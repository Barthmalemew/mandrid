mod db;
mod chunker;
mod task;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use fastembed::{TextEmbedding, TextRerank};
use futures::TryStreamExt;
use globset::{Glob, GlobSet, GlobSetBuilder};
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::index::Index;
use lancedb::index::scalar::FtsIndexBuilder;
use serde::{Deserialize, Serialize};

use arrow_array::RecordBatch;
use ignore::WalkBuilder;
use notify::{Watcher, RecursiveMode, Config, RecommendedWatcher};
use std::sync::mpsc::channel;
use lsp_server::{Connection, Message};
use lsp_types::{
    ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind,
    TextDocumentSyncOptions, notification::{DidChangeTextDocument, Notification},
};

use axum::{
    extract::Query,
    extract::State,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use tower_http::cors::CorsLayer;

use tokio::sync::Mutex;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use crate::chunker::*;
use crate::db::*;
use crate::task::*;

const DEFAULT_DB_DIR: &str = ".mem_db";
const AUTO_DIR_NAME: &str = ".mem_auto";
const TRACE_OUTPUT_MAX_CHARS: usize = 8000;

#[derive(ValueEnum, Clone, Debug)]
enum ContextScope {
    Session,
    Project,
}

#[derive(ValueEnum, Clone, Debug)]
enum AskScope {
    All,
    Code,
    Episodic,
}

#[derive(ValueEnum, Clone, Debug)]
enum AutoEvent {
    GitCommit,
    Command,
    FileChange,
}

fn find_project_root() -> Option<PathBuf> {
    let mut curr = std::env::current_dir().ok()?;
    loop {
        if curr.join(".git").exists() || curr.join(DEFAULT_DB_DIR).exists() {
            return Some(curr);
        }
        if !curr.pop() {
            break;
        }
    }
    None
}

fn register_project(path: &Path) -> Result<()> {
    let config_dir = dirs::config_dir()
        .map(|p| p.join("mandrid"))
        .unwrap_or_else(|| PathBuf::from(".mandrid_config"));
    
    if !config_dir.exists() {
        let _ = fs::create_dir_all(&config_dir);
    }
    
    let registry_path = config_dir.join("projects.json");
    let mut projects: Vec<String> = if registry_path.exists() {
        serde_json::from_str(&fs::read_to_string(&registry_path).unwrap_or_else(|_| "[]".to_string())).unwrap_or_default()
    } else {
        Vec::new()
    };
    
    let abs_path = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let path_str = abs_path.to_string_lossy().to_string();
    if !projects.contains(&path_str) {
        projects.push(path_str);
        let _ = fs::write(registry_path, serde_json::to_string_pretty(&projects)?);
    }
    Ok(())
}

fn get_registered_projects() -> Vec<PathBuf> {
    let config_dir = dirs::config_dir().map(|p| p.join("mandrid"));
    if let Some(dir) = config_dir {
        let registry_path = dir.join("projects.json");
        if registry_path.exists() {
            if let Ok(content) = fs::read_to_string(registry_path) {
                if let Ok(projects) = serde_json::from_str::<Vec<String>>(&content) {
                    return projects.into_iter().map(PathBuf::from).filter(|p| p.exists()).collect();
                }
            }
        }
    }
    Vec::new()
}

#[derive(Parser, Debug)]
#[command(name = "mem", version, about = "Mandrid (mem) - local persistent memory")]
struct Cli {
    #[arg(long, env = "MEM_DB_PATH")]
    db_path: Option<PathBuf>,

    #[arg(long, env = "MEM_CACHE_DIR")]
    cache_dir: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Manually save a thought to memory.
    Save {
        text: String,
        #[arg(long, default_value = "manual")]
        tag: String,
        #[arg(long)]
        session: Option<String>,
    },

    /// Record an agent interaction (reasoning trace/decision).
    Log {
        /// What was done or decided
        text: String,
        /// Optional context tag (e.g. "auth-refactor")
        #[arg(long, default_value = "interaction")]
        tag: String,

        /// Memory type to store (e.g. interaction, trace)
        #[arg(long, default_value = "interaction")]
        memory_type: String,

        /// Optional status (e.g. success, failure)
        #[arg(long, default_value = "n/a")]
        status: String,

        /// Optional file path field (defaults based on memory type)
        #[arg(long)]
        file_path: Option<String>,

        /// Optional name field (e.g. command name)
        #[arg(long)]
        name: Option<String>,

        #[arg(long)]
        session: Option<String>,
    },

    /// Record a reasoning trace or strategic plan.
    Think {
        text: String,
        #[arg(long)]
        session: Option<String>,
    },

    /// Capture current context (git diff + reasoning) into memory.
    Capture {
        /// The reasoning or explanation for the current state/changes
        reasoning: String,

        /// Limit captured diff to these paths (repeatable).
        #[arg(long)]
        paths: Vec<PathBuf>,

        /// Truncate captured diff to this many bytes.
        #[arg(long)]
        max_bytes: Option<usize>,

        #[arg(long)]
        session: Option<String>,
    },

    /// Get optimized context for the current session.
    Context {
        #[arg(long)]
        session: Option<String>,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Filter context by a specific file path
        #[arg(long)]
        file: Option<PathBuf>,
        /// Output in a more human-friendly format
        #[arg(long)]
        human: bool,
        /// Reduce token usage by truncating each entry.
        #[arg(long)]
        compact: bool,
        /// Show only a tiny 1-line summary of the current state.
        #[arg(long)]
        summary: bool,
        /// Context scope (session-only or project-wide).
        #[arg(long, value_enum, default_value = "project")]
        scope: ContextScope,
        /// Maximum estimated tokens for the output.
        #[arg(long)]
        token_budget: Option<usize>,
        /// Include code chunks in the returned context.
        #[arg(long)]
        include_code: bool,
        /// Output in machine-readable JSON format
        #[arg(long)]
        json: bool,
    },

    /// Compress old episodic memories to reduce noise.
    Compress {
        /// Session to compress (default: all)
        #[arg(long)]
        session: Option<String>,
        /// Minimum age in days
        #[arg(long, default_value_t = 1)]
        days: u64,
    },

    /// Permanently delete old memories by type.
    Prune {
        /// Minimum age in days
        #[arg(long, default_value_t = 30)]
        days: u64,

        /// Memory types to prune (comma-separated). Defaults to trace,auto.
        #[arg(long, value_delimiter = ',')]
        types: Vec<String>,

        /// Session to prune (default: all)
        #[arg(long)]
        session: Option<String>,

        /// Keep failure traces (status = failure)
        #[arg(long, default_value_t = true)]
        keep_failures: bool,

        /// Show counts without deleting
        #[arg(long)]
        dry_run: bool,
    },

    /// Manage high-level development tasks.
    Task {
        #[command(subcommand)]
        subcommand: TaskCommand,
    },

    /// Initialize Mandrid in the current directory.
    Init {
        /// Role of the AI agent (programmer or assistant)
        #[arg(long, default_value = "programmer")]
        role: String,

        /// Optional Project ID (generated if not provided)
        #[arg(long)]
        id: Option<String>,
    },

    /// Rebuild the local database after a significant Mandrid update.
    ///
    /// This backs up `.mem_db`, creates a fresh DB, optionally re-imports
    /// episodic memories (thoughts/traces/tasks), and optionally re-learns code.
    Rebuild {
        /// Re-import non-code memories from the backup DB.
        #[arg(long, default_value_t = true)]
        preserve_episodic: bool,

        /// Skip re-learning the code index.
        #[arg(long)]
        skip_learn: bool,

        /// Root dir for code re-learning (default: .)
        #[arg(long, default_value = ".")]
        root_dir: PathBuf,

        #[arg(long, default_value_t = 4)]
        concurrency: usize,
    },

    /// Diagnose Mandrid DB issues and suggest fixes.
    Doctor,

    /// Ingest a single code file.
    Digest {
        file_path: PathBuf,
    },

    /// Recursively scan and memorize all code in a folder.
    Learn {
        #[arg(default_value = ".")]
        root_dir: PathBuf,

        #[arg(long, default_value_t = 4)]
        concurrency: usize,
    },

    /// Find relevant memories or code chunks.
    Ask {
        question: String,

        #[arg(long = "json")]
        json_output: bool,

        #[arg(long, default_value_t = 3)]
        limit: usize,

        /// Disable Hybrid Search (use vector only)
        #[arg(long)]
        vector_only: bool,

        /// Restrict search scope.
        #[arg(long, value_enum, default_value = "all")]
        scope: AskScope,

        /// Boost results relevant to the current active task
        #[arg(long)]
        task_aware: bool,

        /// Enable Cross-Encoder Reranking for hyper-precision
        #[arg(long)]
        rerank: bool,
    },

    /// Build a tiny, token-budgeted context pack for agent runners.
    Pack {
        /// Input/query string used for retrieval.
        #[arg(allow_hyphen_values = true)]
        input: String,

        #[arg(long)]
        session: Option<String>,

        /// Context scope (session-only or project-wide).
        #[arg(long, value_enum, default_value = "session")]
        scope: ContextScope,

        /// Maximum estimated tokens for the output.
        #[arg(long, default_value_t = 800)]
        token_budget: usize,

        /// Number of episodic memories to include.
        #[arg(long, default_value_t = 2)]
        k_episodic: usize,

        /// Number of code chunks to include (requires --include-code).
        #[arg(long, default_value_t = 2)]
        k_code: usize,

        /// Include code chunks (memory_type=code) in the pack.
        #[arg(long)]
        include_code: bool,

        /// Include long-form captured reasoning entries (memory_type=reasoning).
        /// Off by default to keep packs small.
        #[arg(long)]
        include_reasoning: bool,

        /// Only include episodic memories newer than this many days.
        #[arg(long)]
        max_age_days: Option<u64>,

        /// Include only these episodic memory types (comma-separated).
        #[arg(long, value_delimiter = ',')]
        include_types: Vec<String>,

        /// Exclude these episodic memory types (comma-separated).
        #[arg(long, value_delimiter = ',')]
        exclude_types: Vec<String>,

        /// Per-type caps for episodic memories (comma-separated, e.g. trace=2,thought=1)
        #[arg(long, value_delimiter = ',', env = "MEM_PACK_TYPE_CAPS")]
        type_caps: Vec<String>,

        /// Enable Cross-Encoder reranking (slower; optional).
        #[arg(long)]
        rerank: bool,

        /// Disable Hybrid Search (use vector only).
        #[arg(long)]
        vector_only: bool,

        /// Output in machine-readable JSON format.
        #[arg(long)]
        json: bool,
    },

    /// Inspect the local LanceDB memory store.
    Debug {
        #[arg(long, default_value_t = 3)]
        limit: usize,
    },

    /// Show a high-level map of the codebase architecture.
    Map,

    /// Watch the filesystem for changes and automatically re-index files.
    Watch {
        #[arg(default_value = ".")]
        root_dir: PathBuf,
    },

    /// Show all symbols (functions, structs, etc.) found in the codebase.
    Symbols {
        /// Filter by name or file
        #[arg(short, long)]
        query: Option<String>,

        #[arg(long)]
        json: bool,
    },

    /// Automatically sync git history into reasoning traces.
    SyncGit {
        #[arg(long, default_value_t = 10)]
        commits: usize,
    },

    /// Calculate the "Blast Radius" of a symbol (who depends on this?).
    Impact {
        symbol: String,
        #[arg(long, default_value_t = 2)]
        depth: usize,

        #[arg(long)]
        json: bool,
    },

    /// Launch the Mandrid Web Dashboard.
    Serve {
        #[arg(long, default_value_t = 3000)]
        port: u16,

        /// Run in the background as a daemon.
        #[arg(long)]
        background: bool,
    },

    /// Stop the background daemon if running.
    Stop,

    /// Generate shell hooks for automated command capture.
    Hook {
        /// Shell type (zsh, bash, powershell)
        #[arg(default_value = "zsh")]
        shell: String,
    },

    /// Deterministic auto-memory (LLM-free).
    Auto {
        #[command(subcommand)]
        subcommand: AutoCommand,
    },

    /// Propose a fix for the most recent failure in memory.
    Fix {
        #[arg(long)]
        session: Option<String>,
    },

    /// Check for potential risks based on previous failures.
    CheckRisk {
        command_text: String,
        #[arg(long)]
        session: Option<String>,
    },

    /// Explain why a symbol exists or was changed (Chesterton's Fence).
    Why {
        symbol: String,
        #[arg(long)]
        session: Option<String>,
    },

    /// Start a minimal LSP server to receive real-time code changes.
    Lsp,

    /// Sense architectural patterns or smells in the codebase.
    Sense {
        #[arg(default_value = "dead-code")]
        mode: String,
    },

    /// Tools schema for AI agents.
    Tools {
        /// Format (json, markdown)
        #[arg(long, default_value = "markdown")]
        format: String,
    },

    /// Execute a command and automatically record its output and result to memory.
    Run {
        #[arg(trailing_var_arg = true)]
        command: Vec<String>,
        #[arg(long)]
        session: Option<String>,
        /// Custom reasoning or label for this execution
        #[arg(long)]
        note: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
enum AutoCommand {
    /// Initialize deterministic auto-memory config and queue.
    Init {
        /// Overwrite existing config if present.
        #[arg(long)]
        force: bool,
    },

    /// Run an auto-memory event and queue/approve the entry.
    Run {
        #[arg(long, value_enum)]
        event: AutoEvent,

        #[arg(long)]
        session: Option<String>,

        /// Command string for command events.
        #[arg(long)]
        cmd: Option<String>,

        /// Exit status for command events.
        #[arg(long)]
        status: Option<i32>,

        /// Duration in milliseconds for command events.
        #[arg(long)]
        duration_ms: Option<u64>,

        /// Optional note for command events.
        #[arg(long)]
        note: Option<String>,

        /// Commit hash for git_commit events (default: HEAD).
        #[arg(long)]
        commit: Option<String>,

        /// Paths for file_change events.
        #[arg(long)]
        paths: Vec<PathBuf>,
    },

    /// Show pending auto-memory entries.
    Status,

    /// Approve queued auto-memory entries.
    Approve {
        /// Approve all queued entries.
        #[arg(long)]
        all: bool,

        /// Approve a specific entry by id.
        #[arg(long)]
        id: Option<String>,
    },

    /// Reject queued auto-memory entries.
    Reject {
        /// Reject all queued entries.
        #[arg(long)]
        all: bool,

        /// Reject a specific entry by id.
        #[arg(long)]
        id: Option<String>,
    },

    /// Install or remove auto-memory hooks.
    Hook {
        #[arg(value_enum)]
        action: AutoHookAction,
    },
}

#[derive(ValueEnum, Clone, Debug)]
enum AutoHookAction {
    Install,
    Uninstall,
}

#[derive(Debug, serde::Serialize)]
struct PackJsonItem {
    tag: Option<String>,
    memory_type: String,
    file_path: Option<String>,
    line_start: Option<u32>,
    session_id: Option<String>,
    name: Option<String>,
    created_at: u64,
    text: String,
}

#[derive(Debug, serde::Serialize)]
struct PackJsonPayload {
    role: String,
    session_id: String,
    scope: String,
    token_budget: usize,
    input: String,
    active_task_title: Option<String>,
    last_failure: Option<String>,
    episodic: Vec<PackJsonItem>,
    code: Vec<PackJsonItem>,
    pack: String,
}

#[derive(Debug, serde::Serialize)]
struct AskJsonResult {
    rank: usize,
    tag: Option<String>,
    text: String,
    memory_type: String,
    file_path: Option<String>,
    line_start: Option<u32>,
    created_at: u64,
}

#[derive(Debug, serde::Serialize)]
struct AskJsonPayload {
    question: String,
    results: Vec<AskJsonResult>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
struct AutoConfig {
    filters: AutoFilters,
    ttl: AutoTtl,
    review: AutoReview,
}

impl Default for AutoConfig {
    fn default() -> Self {
        Self {
            filters: AutoFilters::default(),
            ttl: AutoTtl::default(),
            review: AutoReview::default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
struct AutoFilters {
    include_paths: Vec<String>,
    exclude_paths: Vec<String>,
    include_commands: Vec<String>,
    max_file_kb: u64,
    max_diff_lines: u64,
}

impl Default for AutoFilters {
    fn default() -> Self {
        Self {
            include_paths: Vec::new(),
            exclude_paths: vec![
                "target/**".to_string(),
                ".mem_db/**".to_string(),
                ".mem_auto/**".to_string(),
                "node_modules/**".to_string(),
                "Cargo.lock".to_string(),
            ],
            include_commands: vec![
                "cargo check".to_string(),
                "cargo test".to_string(),
                "cargo build".to_string(),
                "nix develop".to_string(),
                "npm test".to_string(),
                "pnpm test".to_string(),
                "go test".to_string(),
            ],
            max_file_kb: 256,
            max_diff_lines: 500,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
struct AutoTtl {
    default_days: u64,
    promote_on_reference: bool,
}

impl Default for AutoTtl {
    fn default() -> Self {
        Self {
            default_days: 30,
            promote_on_reference: true,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
struct AutoReview {
    auto_approve: bool,
}

impl Default for AutoReview {
    fn default() -> Self {
        Self { auto_approve: false }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct AutoFact {
    key: String,
    value: String,
    path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct AutoDiffStats {
    added: u64,
    deleted: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(default)]
struct AutoProvenance {
    git: Option<AutoGitProvenance>,
    command: Option<AutoCommandProvenance>,
    file_change: Option<AutoFileChangeProvenance>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct AutoGitProvenance {
    commit: String,
    branch: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct AutoCommandProvenance {
    cmd: String,
    status: i32,
    duration_ms: Option<u64>,
    note: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct AutoFileChangeProvenance {
    paths: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct AutoEntry {
    id: String,
    entry_type: String,
    source: String,
    project_id: String,
    host: String,
    session: String,
    timestamp: u64,
    summary: String,
    facts: Vec<AutoFact>,
    paths: Vec<String>,
    diff_stats: Option<AutoDiffStats>,
    ttl_days: u64,
    importance: f32,
    approved: bool,
    provenance: AutoProvenance,
}

#[derive(Clone)]
struct ServeState {
    active_db_path: Arc<Mutex<PathBuf>>,
    cache_dir: Option<PathBuf>,
    embedder: Arc<Mutex<TextEmbedding>>,
    reranker: Arc<Mutex<Option<TextRerank>>>,
}

#[derive(Debug, Deserialize)]
struct MemoriesQuery {
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, serde::Serialize)]
struct LogRequest {
    text: String,
    tag: String,
    memory_type: String,
    status: String,
    file_path: Option<String>,
    name: Option<String>,
    session: Option<String>,
}

#[derive(Debug, serde::Serialize)]
struct StatusResponse {
    status: String,
    version: String,
    db_path: String,
}

fn get_ignore_matcher(root: &Path) -> ignore::gitignore::Gitignore {
    let mut builder = ignore::gitignore::GitignoreBuilder::new(root);
    let gitignore_path = root.join(".gitignore");
    if gitignore_path.exists() {
        builder.add(gitignore_path);
    }
    // Also ignore Mandrid's own DB
    builder.add_line(None, ".mem_db/").unwrap();
    builder.build().unwrap()
}

fn is_ignored(matcher: &ignore::gitignore::Gitignore, path: &Path) -> bool {
    matcher.matched(path, false).is_ignore()
}

#[derive(Debug, Deserialize, serde::Serialize)]
struct SearchQuery {
    q: String,
    limit: Option<usize>,
    rerank: Option<bool>,
    vector_only: Option<bool>,
}

async fn is_server_alive(port: u16) -> bool {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(100))
        .build()
        .unwrap();
    client.get(format!("http://localhost:{}/api/status", port))
        .send()
        .await
        .is_ok()
}

fn get_hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}

fn get_origin_tag(tag: &str) -> String {
    format!("[{}] {}", get_hostname(), tag)
}

fn escape_lancedb_str(s: &str) -> String {
    s.replace('\'', "''")
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let mut it = s.chars();
    let mut out = String::new();
    for _ in 0..max_chars {
        if let Some(c) = it.next() {
            out.push(c);
        } else {
            return out;
        }
    }
    out.push_str("...");
    out
}

fn truncate_for_trace(text: &str, max_chars: usize) -> (String, bool) {
    if max_chars == 0 {
        return (String::new(), !text.is_empty());
    }
    let count = text.chars().count();
    if count > max_chars {
        (truncate_chars(text, max_chars), true)
    } else {
        (text.to_string(), false)
    }
}

fn normalize_memory_types(types: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for t in types {
        let trimmed = t.trim().to_lowercase();
        if trimmed.is_empty() {
            continue;
        }
        if seen.insert(trimmed.clone()) {
            out.push(trimmed);
        }
    }
    out
}

fn parse_type_caps(items: &[String]) -> HashMap<String, usize> {
    let mut out = HashMap::new();
    for raw in items {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.splitn(2, '=');
        let key = parts.next().unwrap_or("").trim().to_lowercase();
        let value = parts.next().unwrap_or("").trim();
        if key.is_empty() || value.is_empty() {
            continue;
        }
        if let Ok(cap) = value.parse::<usize>() {
            out.insert(key, cap);
        }
    }
    out
}

fn auto_dir(project_root: &Path) -> PathBuf {
    project_root.join(AUTO_DIR_NAME)
}

fn auto_config_path(project_root: &Path) -> PathBuf {
    auto_dir(project_root).join("config.toml")
}

fn auto_queue_path(project_root: &Path) -> PathBuf {
    auto_dir(project_root).join("queue.jsonl")
}

fn auto_gitignore_path(project_root: &Path) -> PathBuf {
    auto_dir(project_root).join(".gitignore")
}

fn ensure_auto_dir(project_root: &Path) -> Result<PathBuf> {
    let dir = auto_dir(project_root);
    if !dir.exists() {
        fs::create_dir_all(&dir)?;
    }
    Ok(dir)
}

fn load_auto_config(project_root: &Path) -> Result<AutoConfig> {
    let path = auto_config_path(project_root);
    if !path.exists() {
        return Ok(AutoConfig::default());
    }
    let raw = fs::read_to_string(&path)?;
    let parsed: AutoConfig = toml::from_str(&raw)?;
    Ok(parsed)
}

fn write_auto_config(project_root: &Path, config: &AutoConfig) -> Result<()> {
    let path = auto_config_path(project_root);
    let body = toml::to_string_pretty(config)?;
    fs::write(path, body)?;
    Ok(())
}

fn load_auto_queue(project_root: &Path) -> Result<Vec<AutoEntry>> {
    let path = auto_queue_path(project_root);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut entries = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<AutoEntry>(&line) {
            entries.push(entry);
        }
    }
    Ok(entries)
}

fn write_auto_queue(project_root: &Path, entries: &[AutoEntry]) -> Result<()> {
    let path = auto_queue_path(project_root);
    let mut file = fs::File::create(path)?;
    for entry in entries {
        let line = serde_json::to_string(entry)?;
        writeln!(file, "{}", line)?;
    }
    Ok(())
}

fn append_auto_queue(project_root: &Path, entry: &AutoEntry) -> Result<()> {
    let path = auto_queue_path(project_root);
    let mut file = fs::OpenOptions::new().create(true).append(true).open(path)?;
    let line = serde_json::to_string(entry)?;
    writeln!(file, "{}", line)?;
    Ok(())
}

fn build_globset(patterns: &[String]) -> Result<Option<GlobSet>> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let mut builder = GlobSetBuilder::new();
    for p in patterns {
        builder.add(Glob::new(p)?);
    }
    Ok(Some(builder.build()?))
}

fn path_allowed(path: &Path, include: Option<&GlobSet>, exclude: Option<&GlobSet>) -> bool {
    if let Some(ex) = exclude {
        if ex.is_match(path) {
            return false;
        }
    }
    if let Some(inc) = include {
        return inc.is_match(path);
    }
    true
}

fn command_allowed(cmd: &str, include: &[String]) -> bool {
    if include.is_empty() {
        return true;
    }
    let trimmed = cmd.trim_start();
    include.iter().any(|p| trimmed.starts_with(p))
}

fn get_project_id(project_root: &Path) -> String {
    let id_path = project_root.join(".mandrid_id");
    if let Ok(raw) = fs::read_to_string(id_path) {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    "unknown".to_string()
}

fn make_auto_entry_id(prefix: &str, seed: &str, timestamp: u64) -> String {
    let mut clean_seed: String = seed.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    if clean_seed.is_empty() {
        clean_seed = "entry".to_string();
    }
    if clean_seed.len() > 12 {
        clean_seed.truncate(12);
    }
    format!("auto-{}-{}-{}", prefix, timestamp, clean_seed)
}

fn format_auto_entry_text(entry: &AutoEntry) -> String {
    let mut out = String::new();
    out.push_str(&format!("Auto memory ({})\n", entry.source));
    out.push_str(&format!("Summary: {}\n", entry.summary));
    if let Some(stats) = entry.diff_stats.as_ref() {
        out.push_str(&format!("Diff: +{} -{}\n", stats.added, stats.deleted));
    }
    if !entry.paths.is_empty() {
        out.push_str("Paths:\n");
        let max_paths = 12;
        for path in entry.paths.iter().take(max_paths) {
            out.push_str(&format!("- {}\n", path));
        }
        if entry.paths.len() > max_paths {
            out.push_str(&format!("... ({} more)\n", entry.paths.len() - max_paths));
        }
    }
    if !entry.facts.is_empty() {
        out.push_str("Facts:\n");
        for fact in &entry.facts {
            if let Some(path) = &fact.path {
                out.push_str(&format!("- {}: {} ({})\n", fact.key, fact.value, path));
            } else {
                out.push_str(&format!("- {}: {}\n", fact.key, fact.value));
            }
        }
    }
    if let Some(git) = &entry.provenance.git {
        out.push_str(&format!("Commit: {} ({})\n", git.commit, git.branch));
    }
    if let Some(cmd) = &entry.provenance.command {
        out.push_str(&format!("Command: {} (status {})\n", cmd.cmd, cmd.status));
    }
    out.trim_end().to_string()
}

fn auto_entry_name(entry: &AutoEntry) -> String {
    if let Some(git) = &entry.provenance.git {
        return git.commit.chars().take(8).collect();
    }
    if let Some(cmd) = &entry.provenance.command {
        return cmd
            .cmd
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string();
    }
    "".to_string()
}

fn auto_entry_file_path(entry: &AutoEntry) -> String {
    if entry.paths.len() == 1 {
        return entry.paths[0].clone();
    }
    "auto".to_string()
}

async fn run_git(project_root: &Path, args: Vec<String>) -> Result<String> {
    let output = tokio::process::Command::new("git")
        .args(args)
        .current_dir(project_root)
        .output()
        .await?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        anyhow::bail!("git command failed: {}", stderr);
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn parse_numstat(output: &str) -> (u64, u64) {
    let mut added = 0u64;
    let mut deleted = 0u64;
    for line in output.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 3 {
            continue;
        }
        if parts[0] != "-" {
            added += parts[0].parse::<u64>().unwrap_or(0);
        }
        if parts[1] != "-" {
            deleted += parts[1].parse::<u64>().unwrap_or(0);
        }
    }
    (added, deleted)
}

fn extract_quoted_value(line: &str) -> Option<String> {
    let start = line.find('"')?;
    let rest = &line[start + 1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn parse_version_from_diff(diff: &str) -> Option<String> {
    for line in diff.lines() {
        if !line.starts_with('+') || line.starts_with("+++") {
            continue;
        }
        let trimmed = line.trim_start_matches('+').trim();
        if trimmed.starts_with("version") || trimmed.starts_with("\"version\"") {
            if let Some(value) = extract_quoted_value(trimmed) {
                return Some(value);
            }
        }
    }
    None
}

async fn detect_version_from_diff(
    project_root: &Path,
    commit: &str,
    path: &str,
) -> Result<Option<String>> {
    let diff = run_git(
        project_root,
        vec![
            "show".to_string(),
            "-1".to_string(),
            "--unified=0".to_string(),
            commit.to_string(),
            "--".to_string(),
            path.to_string(),
        ],
    )
    .await;
    if let Ok(diff) = diff {
        return Ok(parse_version_from_diff(&diff));
    }
    Ok(None)
}

async fn build_git_commit_entry(
    project_root: &Path,
    config: &AutoConfig,
    session_id: &str,
    commit_override: Option<String>,
) -> Result<Option<AutoEntry>> {
    let commit = commit_override.unwrap_or_else(|| "HEAD".to_string());
    let full_hash = run_git(project_root, vec!["rev-parse".to_string(), commit.clone()])
        .await?
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    if full_hash.is_empty() {
        return Ok(None);
    }
    let short_hash: String = full_hash.chars().take(8).collect();
    let show = run_git(
        project_root,
        vec![
            "show".to_string(),
            "-1".to_string(),
            "--name-status".to_string(),
            "--pretty=format:%ct%n%s".to_string(),
            commit.clone(),
        ],
    )
    .await?;

    let mut lines = show.lines();
    let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
    let timestamp = lines
        .next()
        .and_then(|l| l.trim().parse::<u64>().ok())
        .unwrap_or(now);
    let subject = lines.next().unwrap_or("commit").trim().to_string();

    let mut raw_paths = Vec::new();
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.is_empty() {
            continue;
        }
        let status = parts[0];
        let path = if status.starts_with('R') && parts.len() >= 3 {
            parts[2]
        } else if parts.len() >= 2 {
            parts[1]
        } else {
            continue;
        };
        raw_paths.push(path.to_string());
    }

    if raw_paths.is_empty() {
        return Ok(None);
    }

    let include = build_globset(&config.filters.include_paths)?;
    let exclude = build_globset(&config.filters.exclude_paths)?;

    let mut filtered_paths = Vec::new();
    for path in raw_paths {
        let rel = PathBuf::from(&path);
        if path_allowed(&rel, include.as_ref(), exclude.as_ref()) {
            filtered_paths.push(path);
        }
    }

    if filtered_paths.is_empty() {
        return Ok(None);
    }

    let mut final_paths = Vec::new();
    for path in filtered_paths {
        let full = project_root.join(&path);
        if full.exists() {
            if let Ok(meta) = fs::metadata(&full) {
                let size_kb = meta.len() / 1024;
                if config.filters.max_file_kb > 0 && size_kb > config.filters.max_file_kb {
                    continue;
                }
            }
        }
        final_paths.push(path);
    }

    if final_paths.is_empty() {
        return Ok(None);
    }

    let numstat = run_git(
        project_root,
        vec![
            "show".to_string(),
            "-1".to_string(),
            "--numstat".to_string(),
            "--pretty=format:".to_string(),
            commit.clone(),
        ],
    )
    .await?
    .to_string();
    let (added, deleted) = parse_numstat(&numstat);
    if config.filters.max_diff_lines > 0 && added + deleted > config.filters.max_diff_lines {
        return Ok(None);
    }

    let mut facts = vec![
        AutoFact {
            key: "commit".to_string(),
            value: short_hash.clone(),
            path: None,
        },
        AutoFact {
            key: "files_changed".to_string(),
            value: final_paths.len().to_string(),
            path: None,
        },
    ];
    if added + deleted > 0 {
        facts.push(AutoFact {
            key: "diff".to_string(),
            value: format!("+{} -{}", added, deleted),
            path: None,
        });
    }

    if final_paths.iter().any(|p| p == "Cargo.toml") {
        if let Ok(Some(version)) = detect_version_from_diff(project_root, &commit, "Cargo.toml").await {
            facts.push(AutoFact {
                key: "version".to_string(),
                value: version,
                path: Some("Cargo.toml".to_string()),
            });
        }
    }
    if final_paths.iter().any(|p| p == "package.json") {
        if let Ok(Some(version)) = detect_version_from_diff(project_root, &commit, "package.json").await {
            facts.push(AutoFact {
                key: "version".to_string(),
                value: version,
                path: Some("package.json".to_string()),
            });
        }
    }

    let summary = format!(
        "{} ({}) | files: {} | diff: +{} -{}",
        subject,
        short_hash,
        final_paths.len(),
        added,
        deleted
    );

    let mut importance = 0.4;
    if final_paths.iter().any(|p| {
        p.ends_with("Cargo.toml")
            || p.ends_with("package.json")
            || p.ends_with("flake.nix")
            || p.ends_with("go.mod")
            || p.ends_with("pyproject.toml")
    }) {
        importance += 0.3;
    }
    if added + deleted >= 200 {
        importance += 0.2;
    }
    if final_paths.len() >= 10 {
        importance += 0.1;
    }
    if importance > 1.0 {
        importance = 1.0;
    }

    let branch = run_git(
        project_root,
        vec![
            "rev-parse".to_string(),
            "--abbrev-ref".to_string(),
            "HEAD".to_string(),
        ],
    )
    .await
    .ok()
    .and_then(|s| s.lines().next().map(|l| l.trim().to_string()))
    .unwrap_or_else(|| "unknown".to_string());

    Ok(Some(AutoEntry {
        id: make_auto_entry_id("git", &short_hash, timestamp),
        entry_type: "auto".to_string(),
        source: "git_commit".to_string(),
        project_id: get_project_id(project_root),
        host: get_hostname(),
        session: session_id.to_string(),
        timestamp,
        summary,
        facts,
        paths: final_paths,
        diff_stats: Some(AutoDiffStats { added, deleted }),
        ttl_days: config.ttl.default_days,
        importance,
        approved: false,
        provenance: AutoProvenance {
            git: Some(AutoGitProvenance {
                commit: full_hash,
                branch,
            }),
            command: None,
            file_change: None,
        },
    }))
}

fn build_command_entry(
    project_root: &Path,
    config: &AutoConfig,
    session_id: &str,
    cmd: String,
    status: i32,
    duration_ms: Option<u64>,
    note: Option<String>,
) -> Result<Option<AutoEntry>> {
    let cmd_trimmed = cmd.trim();
    if cmd_trimmed.is_empty() {
        return Ok(None);
    }
    if !command_allowed(cmd_trimmed, &config.filters.include_commands) {
        return Ok(None);
    }

    let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
    let mut facts = vec![
        AutoFact {
            key: "command".to_string(),
            value: cmd_trimmed.to_string(),
            path: None,
        },
        AutoFact {
            key: "status".to_string(),
            value: status.to_string(),
            path: None,
        },
    ];
    if let Some(duration) = duration_ms {
        facts.push(AutoFact {
            key: "duration_ms".to_string(),
            value: duration.to_string(),
            path: None,
        });
    }

    let mut importance = 0.4;
    let cmd_lower = cmd_trimmed.to_lowercase();
    if cmd_lower.contains("test") || cmd_lower.contains("check") {
        importance += 0.2;
    }
    if status != 0 {
        importance += 0.2;
    }
    if importance > 1.0 {
        importance = 1.0;
    }

    let summary = if let Some(duration) = duration_ms {
        format!("{} | status: {} | {}ms", cmd_trimmed, status, duration)
    } else {
        format!("{} | status: {}", cmd_trimmed, status)
    };

    Ok(Some(AutoEntry {
        id: make_auto_entry_id("cmd", cmd_trimmed, now),
        entry_type: "auto".to_string(),
        source: "command".to_string(),
        project_id: get_project_id(project_root),
        host: get_hostname(),
        session: session_id.to_string(),
        timestamp: now,
        summary,
        facts,
        paths: Vec::new(),
        diff_stats: None,
        ttl_days: config.ttl.default_days,
        importance,
        approved: false,
        provenance: AutoProvenance {
            git: None,
            command: Some(AutoCommandProvenance {
                cmd: cmd_trimmed.to_string(),
                status,
                duration_ms,
                note,
            }),
            file_change: None,
        },
    }))
}

fn build_file_change_entry(
    project_root: &Path,
    config: &AutoConfig,
    session_id: &str,
    paths: Vec<PathBuf>,
) -> Result<Option<AutoEntry>> {
    if paths.is_empty() {
        return Ok(None);
    }

    let include = build_globset(&config.filters.include_paths)?;
    let exclude = build_globset(&config.filters.exclude_paths)?;

    let mut final_paths = Vec::new();
    for path in paths {
        let abs = fs::canonicalize(&path).unwrap_or(path);
        let rel = abs.strip_prefix(project_root).unwrap_or(&abs);
        let rel_path = PathBuf::from(rel);
        if !path_allowed(&rel_path, include.as_ref(), exclude.as_ref()) {
            continue;
        }
        if let Ok(meta) = fs::metadata(&abs) {
            let size_kb = meta.len() / 1024;
            if config.filters.max_file_kb > 0 && size_kb > config.filters.max_file_kb {
                continue;
            }
        }
        final_paths.push(rel_path.to_string_lossy().to_string());
    }

    if final_paths.is_empty() {
        return Ok(None);
    }

    let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
    let summary = format!("File changes: {} paths", final_paths.len());
    let facts = vec![AutoFact {
        key: "files_changed".to_string(),
        value: final_paths.len().to_string(),
        path: None,
    }];

    Ok(Some(AutoEntry {
        id: make_auto_entry_id("file", &final_paths[0], now),
        entry_type: "auto".to_string(),
        source: "file_change".to_string(),
        project_id: get_project_id(project_root),
        host: get_hostname(),
        session: session_id.to_string(),
        timestamp: now,
        summary,
        facts,
        paths: final_paths.clone(),
        diff_stats: None,
        ttl_days: config.ttl.default_days,
        importance: 0.3,
        approved: false,
        provenance: AutoProvenance {
            git: None,
            command: None,
            file_change: Some(AutoFileChangeProvenance { paths: final_paths }),
        },
    }))
}

async fn prune_auto_entries(table: &lancedb::Table, max_age_days: u64) -> Result<()> {
    if max_age_days == 0 {
        return Ok(());
    }
    let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
    let threshold = now.saturating_sub(max_age_days.saturating_mul(86400));
    let filter = format!("memory_type = 'auto' AND created_at < {}", threshold);
    table.delete(&filter).await?;
    Ok(())
}

async fn store_auto_entries(
    db_path: &Path,
    cache_dir: Option<PathBuf>,
    entries: &[AutoEntry],
    prune_days: Option<u64>,
) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    let table = open_or_create_table(db_path).await?;
    if let Some(days) = prune_days {
        if days > 0 {
            let _ = prune_auto_entries(&table, days).await;
        }
    }
    let mut embedder = init_embedder(cache_dir, false)?;
    let mut rows = Vec::new();
    for entry in entries {
        let text = format_auto_entry_text(entry);
        let embedding = embed_prefixed(&mut embedder, "passage", &text)?;
        let row = MemoryRow {
            vector: embedding,
            text,
            tag: get_origin_tag(&format!("auto:{}", entry.source)),
            memory_type: "auto".to_string(),
            file_path: auto_entry_file_path(entry),
            line_start: 0,
            session_id: entry.session.clone(),
            name: auto_entry_name(entry),
            references: serde_json::to_string(&entry.paths).unwrap_or_else(|_| "[]".to_string()),
            depends_on: "[]".to_string(),
            status: "active".to_string(),
            mtime_secs: 0,
            size_bytes: 0,
            created_at: entry.timestamp,
        };
        rows.push(row);
    }
    add_rows(&table, rows).await?;
    Ok(())
}

fn auto_hook_script() -> String {
    "#!/bin/sh\n# mandrid-auto-hook\nif command -v mem >/dev/null 2>&1; then\n  mem auto run --event git_commit >/dev/null 2>&1\nfi\n".to_string()
}

fn resolve_git_hooks_dir(project_root: &Path) -> Result<PathBuf> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--git-path", "hooks"])
        .current_dir(project_root)
        .output()?;
    if !output.status.success() {
        anyhow::bail!("Failed to resolve git hooks directory.");
    }
    let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if raw.is_empty() {
        anyhow::bail!("Failed to resolve git hooks directory.");
    }
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(project_root.join(path))
    }
}

fn install_auto_git_hook(project_root: &Path) -> Result<()> {
    let hooks_dir = resolve_git_hooks_dir(project_root)?;
    fs::create_dir_all(&hooks_dir)?;
    let hook_path = hooks_dir.join("post-commit");
    if hook_path.exists() {
        let existing = fs::read_to_string(&hook_path).unwrap_or_default();
        if existing.contains("mandrid-auto-hook") {
            println!("Auto hook already installed: {}", hook_path.display());
            return Ok(());
        }
        anyhow::bail!(
            "Hook already exists at {}. Remove or edit it to install Mandrid auto hook.",
            hook_path.display()
        );
    }
    let script = auto_hook_script();
    fs::write(&hook_path, script)?;
    #[cfg(unix)]
    {
        let mut perms = fs::metadata(&hook_path)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&hook_path, perms)?;
    }
    println!("Installed git post-commit hook at {}", hook_path.display());
    Ok(())
}

fn uninstall_auto_git_hook(project_root: &Path) -> Result<()> {
    let hooks_dir = resolve_git_hooks_dir(project_root)?;
    let hook_path = hooks_dir.join("post-commit");
    if !hook_path.exists() {
        println!("No post-commit hook found.");
        return Ok(());
    }
    let existing = fs::read_to_string(&hook_path).unwrap_or_default();
    if existing.contains("mandrid-auto-hook") {
        fs::remove_file(&hook_path)?;
        println!("Removed auto hook from {}", hook_path.display());
        return Ok(());
    }
    anyhow::bail!(
        "Post-commit hook at {} does not look like a Mandrid auto hook.",
        hook_path.display()
    );
}

fn scope_to_str(scope: &ContextScope) -> &'static str {
    match scope {
        ContextScope::Session => "session",
        ContextScope::Project => "project",
    }
}

fn to_pack_item(row: &DecodedRow, max_chars: usize, include_location: bool) -> PackJsonItem {
    PackJsonItem {
        tag: row.tag.clone(),
        memory_type: row.memory_type.clone(),
        file_path: if include_location {
            Some(row.file_path.clone())
        } else {
            None
        },
        line_start: if include_location { Some(row.line_start) } else { None },
        session_id: Some(row.session_id.clone()),
        name: if row.name.is_empty() { None } else { Some(row.name.clone()) },
        created_at: row.created_at,
        text: truncate_chars(row.text.trim(), max_chars),
    }
}

fn build_pack_text(
    role: &str,
    session_id: &str,
    scope: &ContextScope,
    token_budget: usize,
    input: &str,
    active_task_title: Option<&str>,
    last_failure: Option<&str>,
    episodic: &[DecodedRow],
    code: &[DecodedRow],
) -> String {
    let budget_chars = token_budget.saturating_mul(4);
    let mut out = String::new();

    // Always include mandatory fields.
    out.push_str("<mem_pack>\n");
    out.push_str(&format!("role: {}\n", role));
    out.push_str(&format!("session: {}\n", session_id));
    out.push_str(&format!("scope: {}\n", scope_to_str(scope)));
    out.push_str(&format!(
        "task: {}\n",
        active_task_title.unwrap_or("none")
    ));
    out.push_str(&format!("last_failure: {}\n", last_failure.unwrap_or("none")));
    out.push_str(&format!("input: {}\n", truncate_chars(input.trim(), 240)));

    // If even the header exceeds the budget, stop here (mandatory content already emitted).
    if out.len() > budget_chars {
        out.push_str("</mem_pack>\n");
        return out;
    }

    let try_push = |s: &str, out: &mut String| -> bool {
        if out.len().saturating_add(s.len()) > budget_chars {
            return false;
        }
        out.push_str(s);
        true
    };

    if !episodic.is_empty() {
        if !try_push("\n[episodic]\n", &mut out) {
            out.push_str("</mem_pack>\n");
            return out;
        }

        for r in episodic {
            let tag = r.tag.as_deref().unwrap_or("unknown");
            let first = r.text.lines().next().unwrap_or("").trim();
            let item = format!(
                "- {} ({}) {}\n",
                tag,
                r.memory_type,
                truncate_chars(first, 220)
            );
            if !try_push(&item, &mut out) {
                out.push_str("\n[...budget exceeded...]\n");
                out.push_str("</mem_pack>\n");
                return out;
            }
        }
    }

    if !code.is_empty() {
        if !try_push("\n[code]\n", &mut out) {
            out.push_str("</mem_pack>\n");
            return out;
        }

        for r in code {
            let name = if r.name.is_empty() { "" } else { &r.name };
            let header = if name.is_empty() {
                format!("- {}:{}\n", r.file_path, r.line_start)
            } else {
                format!("- {}:{} {}\n", r.file_path, r.line_start, name)
            };
            if !try_push(&header, &mut out) {
                out.push_str("\n[...budget exceeded...]\n");
                out.push_str("</mem_pack>\n");
                return out;
            }
            let snippet = truncate_chars(r.text.trim(), 520);
            let body = format!("{}\n", snippet);
            if !try_push(&body, &mut out) {
                out.push_str("\n[...budget exceeded...]\n");
                out.push_str("</mem_pack>\n");
                return out;
            }
        }
    }

    out.push_str("</mem_pack>\n");
    out
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    let project_root = find_project_root().unwrap_or_else(|| std::env::current_dir().unwrap());
    let db_path = cli.db_path.unwrap_or_else(|| project_root.join(DEFAULT_DB_DIR));
    let cache_dir = cli.cache_dir.clone();

    match cli.command {
        Command::Save { text, tag, session } => {
            let mut embedder = init_embedder(cache_dir.clone(), true)?;
            let table = open_or_create_table(&db_path).await?;

            let embedding = embed_prefixed(&mut embedder, "passage", &text)?;
            let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
            let row = MemoryRow {
                vector: embedding,
                text,
                tag: get_origin_tag(&tag),
                memory_type: "manual".to_string(),
                file_path: "manual".to_string(),
                line_start: 0,
                session_id: session.unwrap_or_else(|| "default".to_string()),
                name: String::new(),
                references: "[]".to_string(),
                depends_on: "[]".to_string(),
                status: "n/a".to_string(),
                mtime_secs: 0,
                size_bytes: 0,
                created_at: now,
            };
            add_rows(&table, vec![row]).await?;
            println!("Saved");
        }
        Command::Log { text, tag, memory_type, status, file_path, name, session } => {
            if is_server_alive(3000).await {
                let client = reqwest::Client::new();
                let req = LogRequest {
                    text, tag, memory_type, status, file_path, name, session
                };
                let _ = client.post("http://localhost:3000/api/log")
                    .json(&req)
                    .send()
                    .await;
                println!("Logged (via daemon)");
                return Ok(());
            }

            let mut embedder = init_embedder(cache_dir.clone(), true)?;
            let table = open_or_create_table(&db_path).await?;

            let prefix = if memory_type == "trace" { "trace" } else { "passage" };
            let embedding = embed_prefixed(&mut embedder, prefix, &text)?;
            let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();

            let default_path = if memory_type == "trace" { "terminal" } else { "interaction" };
            let row = MemoryRow {
                vector: embedding,
                text,
                tag: get_origin_tag(&tag),
                memory_type: memory_type.clone(),
                file_path: file_path.unwrap_or_else(|| default_path.to_string()),
                line_start: 0,
                session_id: session.unwrap_or_else(|| "default".to_string()),
                name: name.unwrap_or_default(),
                references: "[]".to_string(),
                depends_on: "[]".to_string(),
                status,
                mtime_secs: 0,
                size_bytes: 0,
                created_at: now,
            };
            add_rows(&table, vec![row]).await?;
            println!("Logged");
        }
        Command::Think { text, session } => {
            let mut embedder = init_embedder(cache_dir.clone(), true)?;
            let table = open_or_create_table(&db_path).await?;

            let embedding = embed_prefixed(&mut embedder, "thought", &text)?;
            let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
            let row = MemoryRow {
                vector: embedding,
                text,
                tag: "thinking".to_string(),
                memory_type: "thought".to_string(),
                file_path: "agent_brain".to_string(),
                line_start: 0,
                session_id: session.unwrap_or_else(|| "default".to_string()),
                name: String::new(),
                references: "[]".to_string(),
                depends_on: "[]".to_string(),
                status: "active".to_string(),
                mtime_secs: 0,
                size_bytes: 0,
                created_at: now,
            };
            add_rows(&table, vec![row]).await?;
            println!("Thought recorded.");
        }
        Command::Compress { session, days } => {
            let table = open_table(&db_path).await?;
            let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
            let threshold = now - (days * 86400);

            let mut filter = format!("created_at < {} AND memory_type IN ('trace', 'git_reasoning', 'thought')", threshold);
            if let Some(s) = session {
                filter.push_str(&format!(" AND session_id = '{}'", s.replace('\'', "''")));
            }

            println!("🗜️ Scanning for memories to compress...");
            let stream = table.query().only_if(filter.clone()).execute().await?;
            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            let rows = decode_rows(&batches, None)?;

            if rows.is_empty() {
                println!("No old episodic memories found.");
                return Ok(());
            }

            // Group by session
            let mut session_groups: HashMap<String, Vec<DecodedRow>> = HashMap::new();
            for r in rows {
                session_groups.entry(r.session_id.clone()).or_default().push(r);
            }

            for (session_id, group) in session_groups {
                let total = group.len();
                let failures = group.iter().filter(|r| r.status == "failure").count();
                let last_success = group.iter().rev().find(|r| r.status == "success");
                
                // Extract unique goals/tasks mentioned in this session
                let mut goals = group.iter()
                    .filter(|r| r.memory_type == "thought")
                    .map(|r| r.text.lines().next().unwrap_or("").to_string())
                    .collect::<Vec<_>>();
                goals.sort();
                goals.dedup();

                let summary_text = format!(
                    "SESSION SUMMARY [{}]: Compressed {} items. Encountered {} failures. Goals addressed: {}. Final result: {}",
                    session_id,
                    total,
                    failures,
                    if goals.is_empty() { "n/a".to_string() } else { goals.join(", ") },
                    last_success.map(|r| r.text.lines().next().unwrap_or("n/a")).unwrap_or("Incomplete")
                );

                // Create the summary
                let mut embedder = init_embedder(cache_dir.clone(), false)?;
                let embedding = embed_prefixed(&mut embedder, "summary", &summary_text)?;
                let summary_row = MemoryRow {
                    vector: embedding,
                    text: summary_text,
                    tag: format!("summary:{}", session_id),
                    memory_type: "summary".to_string(),
                    file_path: "mandrid_archive".to_string(),
                    line_start: 0,
                    session_id: session_id.clone(),
                    name: "compression_job".to_string(),
                    references: "[]".to_string(),
                    depends_on: "[]".to_string(),
                    status: "archived".to_string(),
                    mtime_secs: 0,
                    size_bytes: 0,
                    created_at: now,
                };

                add_rows(&table, vec![summary_row]).await?;
                println!("✅ Created summary for session: {}", session_id);
            }

            // Delete the old noisy rows
            table.delete(&filter).await?;
            println!("🗑️ Deleted old episodic traces.");
        }
        Command::Prune {
            days,
            types,
            session,
            keep_failures,
            dry_run,
        } => {
            let table = open_table(&db_path).await?;
            let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
            let threshold = now.saturating_sub(days.saturating_mul(86400));

            let mut types = normalize_memory_types(&types);
            if types.is_empty() {
                types = vec!["trace".to_string(), "auto".to_string()];
            }

            let mut filter_parts = vec![format!("created_at < {}", threshold)];
            if !types.is_empty() {
                let list = types
                    .iter()
                    .map(|t| format!("'{}'", escape_lancedb_str(t)))
                    .collect::<Vec<_>>()
                    .join(", ");
                filter_parts.push(format!("memory_type IN ({})", list));
            }
            if keep_failures {
                filter_parts.push("status != 'failure'".to_string());
            }
            if let Some(s) = session {
                filter_parts.push(format!(
                    "session_id = '{}'",
                    escape_lancedb_str(&s)
                ));
            }
            let filter = filter_parts.join(" AND ");

            if dry_run {
                let stream = table.query().only_if(filter.clone()).execute().await?;
                let batches: Vec<RecordBatch> = stream.try_collect().await?;
                let rows = decode_rows(&batches, None)?;
                if rows.is_empty() {
                    println!("No memories matched the prune filter.");
                    return Ok(());
                }
                let mut counts: HashMap<String, usize> = HashMap::new();
                for row in rows {
                    *counts.entry(row.memory_type).or_insert(0) += 1;
                }
                println!("Prune dry run (older than {} days):", days);
                for (kind, count) in counts.iter() {
                    println!("- {}: {}", kind, count);
                }
                return Ok(());
            }

            table.delete(&filter).await?;
            println!("Pruned memories older than {} days.", days);
        }
        Command::Capture { reasoning, paths, max_bytes, session } => {
            let mut embedder = init_embedder(cache_dir.clone(), true)?;
            let table = open_or_create_table(&db_path).await?;

            let mut cmd = tokio::process::Command::new("git");
            cmd.arg("diff").arg("HEAD");
            if !paths.is_empty() {
                cmd.arg("--");
                for p in &paths {
                    cmd.arg(p);
                }
            }

            let output = cmd
                .current_dir(&project_root)
                .output()
                .await
                .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_else(|_| "No git diff available".to_string());

            let output = if let Some(max) = max_bytes {
                if output.len() > max {
                    let mut out = output;
                    out.truncate(max);
                    out.push_str("\n\n... [diff truncated] ...\n");
                    out
                } else {
                    output
                }
            } else {
                output
            };

            let full_text = format!("Reasoning: {}\n\nChanges:\n{}", reasoning, output);
            let embedding = embed_prefixed(&mut embedder, "passage", &full_text)?;
            let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();

            let row = MemoryRow {
                vector: embedding,
                text: full_text,
                tag: "capture".to_string(),
                memory_type: "reasoning".to_string(),
                file_path: "git".to_string(),
                line_start: 0,
                session_id: session.unwrap_or_else(|| "default".to_string()),
                name: String::new(),
                references: "[]".to_string(),
                depends_on: "[]".to_string(),
                status: "n/a".to_string(),
                mtime_secs: 0,
                size_bytes: 0,
                created_at: now,
            };
            add_rows(&table, vec![row]).await?;
            println!("Memory captured.");
        }

        Command::Context {
            session,
            limit,
            file,
            human,
            compact,
            summary,
            scope,
            token_budget,
            include_code,
            json,
        } => {
            let table = open_table(&db_path).await.with_context(|| {
                format!(
                    "Database not found at {}. Run `mem init` first.",
                    db_path.display()
                )
            })?;

            let session_id = session.unwrap_or_else(|| "default".to_string());

            // 1. Get Project Role/Config
            let config_stream = table
                .query()
                .only_if("memory_type = 'system_config'")
                .execute()
                .await?;
            let config_batches: Vec<RecordBatch> = config_stream.try_collect().await?;
            let config_rows = decode_rows(&config_batches, None)?;
            let role = config_rows
                .first()
                .map(|r| r.text.as_str())
                .unwrap_or("programmer");

            if summary {
                // Fetch just enough for a summary
                let task_stream = table
                    .query()
                    .only_if("memory_type = 'task' AND status = 'active'")
                    .limit(1)
                    .execute()
                    .await?;
                let task_batches: Vec<RecordBatch> = task_stream.try_collect().await?;
                let active_tasks = decode_rows(&task_batches, None)?;

                let fail_stream = table
                    .query()
                    .only_if("memory_type = 'trace' AND status = 'failure'")
                    .limit(1)
                    .execute()
                    .await?;
                let fail_batches: Vec<RecordBatch> = fail_stream.try_collect().await?;
                let last_fail = decode_rows(&fail_batches, None)?;

                let task_text = active_tasks
                    .first()
                    .map(|t| t.text.chars().take(40).collect::<String>())
                    .unwrap_or_else(|| "none".to_string());
                let fail_text = last_fail
                    .first()
                    .map(|f| f.name.chars().take(20).collect::<String>())
                    .unwrap_or_else(|| "none".to_string());

                println!(
                    "Mandrid: Role={}, Session={}, Task={}, LastFail={}",
                    role, session_id, task_text, fail_text
                );
                return Ok(());
            }

            // 2. Get Active Task(s)
            let task_filter = if matches!(scope, ContextScope::Session) {
                format!(
                    "memory_type = 'task' AND status = 'active' AND session_id = '{}'",
                    session_id.replace('\'', "''")
                )
            } else {
                "memory_type = 'task' AND status = 'active'".to_string()
            };
            let task_stream = table.query().only_if(task_filter).execute().await?;
            let task_batches: Vec<RecordBatch> = task_stream.try_collect().await?;
            let active_tasks = decode_rows(&task_batches, None)?;

            // 3. Get Recent Thoughts
            let thought_filter = if matches!(scope, ContextScope::Session) {
                format!(
                    "memory_type = 'thought' AND status = 'active' AND session_id = '{}'",
                    session_id.replace('\'', "''")
                )
            } else {
                "memory_type = 'thought' AND status = 'active'".to_string()
            };
            let thought_stream = table
                .query()
                .only_if(thought_filter)
                .limit(5)
                .execute()
                .await?;
            let thought_batches: Vec<RecordBatch> = thought_stream.try_collect().await?;
            let active_thoughts = decode_rows(&thought_batches, None)?;

            // 4. Get Recent Failures (Negative Memory)
            let failure_filter = if matches!(scope, ContextScope::Session) {
                format!(
                    "memory_type = 'trace' AND status = 'failure' AND session_id = '{}'",
                    session_id.replace('\'', "''")
                )
            } else {
                "memory_type = 'trace' AND status = 'failure'".to_string()
            };
            let failure_stream = table
                .query()
                .only_if(failure_filter)
                .limit(3)
                .execute()
                .await?;
            let failure_batches: Vec<RecordBatch> = failure_stream.try_collect().await?;
            let recent_failures = decode_rows(&failure_batches, None)?;

            let mut filter = if matches!(scope, ContextScope::Session) {
                format!("session_id = '{}'", session_id.replace('\'', "''"))
            } else {
                format!(
                    "(session_id = '{}' OR memory_type = 'manual' OR memory_type = 'task' OR memory_type = 'thought')",
                    session_id.replace('\'', "''")
                )
            };
            if !include_code {
                filter.push_str(" AND memory_type != 'code'");
            }
            filter.push_str(" AND memory_type != 'system_config'");

            if let Some(ref f) = file {
                let abs_path = fs::canonicalize(f).unwrap_or(f.clone());
                let rel_path = abs_path.strip_prefix(&project_root).unwrap_or(&abs_path);
                filter.push_str(&format!(
                    " AND (file_path = '{}' OR memory_type = 'manual' OR memory_type = 'task' OR memory_type = 'thought')",
                    rel_path.display().to_string().replace('\'', "''")
                ));
            }

            // --- IMPROVED: Task-Relevant Context Injection ---
            // We split the limit: 70% recent, 30% semantically relevant to current goal
            let recent_limit = (limit as f32 * 0.7) as usize;
            let semantic_limit = limit - recent_limit;

            // Pull a bit more than needed, then sort by created_at for true recency.
            let fetch_limit = std::cmp::max(recent_limit.saturating_mul(5), recent_limit);
            let stream = table
                .query()
                .only_if(filter.clone())
                .limit(fetch_limit)
                .execute()
                .await?;
            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            let mut rows = decode_rows(&batches, None)?;
            rows.sort_by(|a, b| b.created_at.cmp(&a.created_at));
            rows.truncate(recent_limit);

            // If we have an active task, pull in semantically relevant historical memories
            if !active_tasks.is_empty() && semantic_limit > 0 {
                let mut embedder = init_embedder(cache_dir.clone(), false)?;
                let task_text = &active_tasks[0].text;
                let task_vec = embed_prefixed(&mut embedder, "query", task_text)?;

                // Search for relevant memories that AREN'T already in the recent list
                let semantic_fetch = semantic_limit + rows.len();
                let semantic_filter = if include_code {
                    "memory_type != 'system_config'".to_string()
                } else {
                    "memory_type != 'code' AND memory_type != 'system_config'".to_string()
                };
                let semantic_results = ask_rrf_filtered(
                    &table,
                    &task_vec,
                    task_text,
                    semantic_fetch,
                    false,
                    &semantic_filter,
                )
                .await?;
                for sr in semantic_results {
                    if !rows.iter().any(|r| r.text == sr.text) && rows.len() < limit {
                        rows.push(sr);
                    }
                }
            }

            // --- FEATURE: Symbolic Definition Splicing ---
            let mut external_definitions = Vec::new();
            if let Some(ref f) = file {
                if let Ok(content) = fs::read_to_string(f) {
                    let chunks = structural_chunk(&content, f);
                    let mut unique_refs = std::collections::HashSet::new();
                    for c in chunks {
                        for r in c.references {
                            unique_refs.insert(r);
                        }
                    }

                    // For each reference, look up its definition in the DB
                    for r in unique_refs {
                        // Avoid recursion or noise: don't look up common keywords
                        if r.len() < 3 {
                            continue;
                        }
                        let ref_filter =
                            format!("memory_type = 'code' AND name = '{}'", r.replace('\'', "''"));
                        let stream = table
                            .query()
                            .only_if(ref_filter)
                            .limit(1)
                            .execute()
                            .await?;
                        let batches: Vec<RecordBatch> = stream.try_collect().await?;
                        let mut defs = decode_rows(&batches, None)?;
                        if let Some(def) = defs.pop() {
                            // Only add if it's from a DIFFERENT file than the one we are focusing on
                            if def.file_path != file.as_ref().unwrap().display().to_string() {
                                external_definitions.push(def);
                            }
                        }
                    }
                }
            }
            // ----------------------------------------------

            if json {
                let payload = serde_json::json!({
                    "role": role,
                    "session_id": session_id,
                    "active_tasks": active_tasks,
                    "active_thoughts": active_thoughts,
                    "recent_failures": recent_failures,
                    "external_definitions": external_definitions,
                    "memories": rows
                });
                println!("{}", serde_json::to_string_pretty(&payload)?);
            } else if human {
                println!("<project_context>");

                println!("=== MANDRID STATE SUMMARY ===");
                println!("Role: {}", role.to_uppercase());
                println!("Active Session: {}", session_id);

                let mut current_tokens = 0;
                let budget = token_budget.unwrap_or(2000);
                let chars_per_token = 4; // conservative estimate

                let mut out_sections = Vec::new();

                if !active_tasks.is_empty() {
                    let mut s = "\n[Active Tasks]\n".to_string();
                    for t in &active_tasks {
                        s.push_str(&format!("- {}: {}\n", t.tag.as_ref().unwrap_or(&"".to_string()), t.text));
                    }
                    out_sections.push((10, s)); // Priority 10 (High)
                }
                if !active_thoughts.is_empty() {
                    let mut s = "\n[🧠 BRAIN DUMP]\n".to_string();
                    for t in &active_thoughts {
                        s.push_str(&format!("- {}\n", t.text));
                    }
                    out_sections.push((8, s));
                }
                if !recent_failures.is_empty() {
                    let mut s = "\n## ⚠️ RECENT FAILURES\n".to_string();
                    for f in &recent_failures {
                        s.push_str(&format!("- {}: {}\n", f.tag.as_ref().unwrap_or(&"unknown".to_string()), f.text.lines().next().unwrap_or("")));
                    }
                    out_sections.push((12, s)); // Highest priority
                }

                if !external_definitions.is_empty() {
                    let mut s = "\n## 🕸️ EXTERNAL SYMBOL DEFINITIONS (Referenced in this file)\n".to_string();
                    for def in &external_definitions {
                        s.push_str(&format!("### {} ({}:{})\n{}\n", def.name, def.file_path, def.line_start, def.text));
                    }
                    out_sections.push((2, s)); // Low priority
                }

                // Add memories
                let mut mem_s = format!("\n## Active Session: {}\n", session_id);
                for row in &rows {
                    if (row.memory_type == "task" || row.memory_type == "thought") && row.status == "active" { continue; }
                    
                    let header = format!("## {} ({})", row.tag.as_ref().unwrap_or(&"unknown".to_string()), row.memory_type);
                    if row.memory_type == "code" {
                        mem_s.push_str(&format!("{} {}:{}\n", header, row.file_path, row.line_start));
                    } else {
                        mem_s.push_str(&format!("{}\n", header));
                    }

                    if compact {
                        let lines: Vec<&str> = row.text.lines().collect();
                        if lines.len() > 12 {
                            for l in &lines[..12] { mem_s.push_str(l); mem_s.push('\n'); }
                            mem_s.push_str(&format!("... [{} lines truncated] ...\n", lines.len() - 12));
                        } else {
                            mem_s.push_str(&row.text);
                            mem_s.push('\n');
                        }
                    } else if row.text.len() > 2000 && (row.memory_type == "trace" || row.memory_type == "reasoning") {
                        let lines: Vec<&str> = row.text.lines().collect();
                        if lines.len() > 40 {
                            for line in &lines[..20] { mem_s.push_str(line); mem_s.push('\n'); }
                            mem_s.push_str(&format!("... [{} lines truncated for token efficiency] ...\n", lines.len() - 40));
                            for line in &lines[lines.len()-20..] { mem_s.push_str(line); mem_s.push('\n'); }
                        } else {
                            mem_s.push_str(&row.text);
                            mem_s.push('\n');
                        }
                    } else {
                        mem_s.push_str(&row.text);
                        mem_s.push('\n');
                    }
                }
                out_sections.push((5, mem_s));

                // Sort by priority (desc)
                out_sections.sort_by(|a, b| b.0.cmp(&a.0));

                for (_prio, content) in out_sections {
                    let section_tokens = content.len() / chars_per_token;
                    if current_tokens + section_tokens > budget && current_tokens > 0 {
                        println!("\n[... Context Budget Exceeded ({} tokens) ...]", budget);
                        break;
                    }
                    print!("{}", content);
                    current_tokens += section_tokens;
                }

                println!("</project_context>");
            }
        }
        Command::Init { role, id } => {
            if !db_path.exists() {
                fs::create_dir_all(&db_path).context("Failed to create .mem_db")?;
            }
            let table = open_or_create_table(&db_path).await?;
            let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
            
            let project_id = id.unwrap_or_else(|| {
                // Try to read from .mandrid_id first
                let id_file = project_root.join(".mandrid_id");
                if id_file.exists() {
                    fs::read_to_string(id_file).unwrap_or_default().trim().to_string()
                } else {
                    let t = SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .map(|d| d.as_nanos())
                        .unwrap_or(0);
                    let new_id = format!("proj_{}", t);
                    let _ = fs::write(id_file, &new_id);
                    new_id
                }
            });

            let mut rows = Vec::new();

            // Store role
            rows.push(MemoryRow {
                vector: vec![0.0; EMBEDDING_DIMS as usize],
                text: role.clone(),
                tag: "system:role".to_string(),
                memory_type: "system_config".to_string(),
                file_path: "config".to_string(),
                line_start: 0,
                session_id: "global".to_string(),
                name: String::new(),
                references: "[]".to_string(),
                depends_on: "[]".to_string(),
                status: "active".to_string(),
                mtime_secs: 0,
                size_bytes: 0,
                created_at: now,
            });
            
            // Store project ID
            rows.push(MemoryRow {
                vector: vec![0.0; EMBEDDING_DIMS as usize],
                text: project_id.clone(),
                tag: "system:project_id".to_string(),
                memory_type: "system_config".to_string(),
                file_path: "config".to_string(),
                line_start: 0,
                session_id: "global".to_string(),
                name: String::new(),
                references: "[]".to_string(),
                depends_on: "[]".to_string(),
                status: "active".to_string(),
                mtime_secs: 0,
                size_bytes: 0,
                created_at: now,
            });

            let _ = table.delete("memory_type = 'system_config'").await;
            add_rows(&table, rows).await?;

            // Setup .gitignore
            let gitignore_path = Path::new(".gitignore");
            let mut current_ignore = if gitignore_path.exists() { fs::read_to_string(gitignore_path)? } else { String::new() };
            if !current_ignore.contains(".mem_db/") {
                current_ignore.push_str("\n.mem_db/\n");
                fs::write(gitignore_path, current_ignore)?;
            }

            // Setup AGENTS.md
            let agents_md_path = Path::new("AGENTS.md");
            if !agents_md_path.exists() {
                let agents_content = format!(r#"# Mandrid Agent Guide
You are an AI agent operating in a terminal environment. Mandrid (`mem`) is your local memory layer.

## How to use Mandrid:
1. **Bootstrap Context:** Always start by running `mem context`. This gives you the project role, active tasks, reasoning history, and recent failures.
2. **Memory Retrieval:** When you need to understand code or find specific logic, use `mem ask "your question" --rerank`.
3. **Reasoning Persistence:** Before taking a major action, use `mem think "your plan"` to store your reasoning.
4. **Impact Analysis:** Before changing a function, use `mem impact <symbol_name>` to see what else might break.
5. **Episodic Capture:** For complex multi-step commands, wrap them in `mem run -- <command>` to ensure the output and success/failure are remembered.

Your current assigned role: **{}**
"#, role);
                fs::write(agents_md_path, agents_content)?;
            }
            println!("Mandrid initialized with role: {}", role);
            let _ = register_project(&project_root);
        }
        Command::Rebuild { preserve_episodic, skip_learn, root_dir, concurrency } => {
            let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
            let base = db_path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| DEFAULT_DB_DIR.to_string());
            let backup_path = db_path.with_file_name(format!("{base}.bak.{now}"));

            let mut recovered_role: Option<String> = None;
            let mut recovered_rows: Vec<DecodedRow> = Vec::new();

            if db_path.exists() {
                println!("Backing up {} -> {}", db_path.display(), backup_path.display());
                fs::rename(&db_path, &backup_path).context("Failed to backup existing .mem_db")?;

                if preserve_episodic {
                    match open_table_existing_unchecked(&backup_path).await {
                        Ok(old_table) => {
                            // Recover role (best-effort)
                            if let Ok(stream) = old_table
                                .query()
                                .only_if("tag = 'system:role'")
                                .limit(1)
                                .execute()
                                .await
                            {
                                let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap_or_default();
                                let rows = decode_rows_relaxed(&batches, None).unwrap_or_default();
                                recovered_role = rows.first().map(|r| r.text.clone());
                            }

                            // Recover all non-code memories (we re-embed them below).
                            if let Ok(stream) = old_table
                                .query()
                                .only_if("memory_type != 'code' AND memory_type != 'system_config'")
                                .execute()
                                .await
                            {
                                let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap_or_default();
                                recovered_rows = decode_rows_relaxed(&batches, None).unwrap_or_default();
                            }
                        }
                        Err(e) => {
                            eprintln!("Warning: failed to open backup DB for recovery: {e}");
                        }
                    }
                }
            }

            fs::create_dir_all(&db_path).context("Failed to create new .mem_db")?;
            let table = open_or_create_table(&db_path).await?;

            // Determine project_id (prefer .mandrid_id)
            let id_file = project_root.join(".mandrid_id");
            let project_id = if id_file.exists() {
                fs::read_to_string(&id_file).unwrap_or_default().trim().to_string()
            } else {
                let t = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0);
                let new_id = format!("proj_{}", t);
                let _ = fs::write(&id_file, &new_id);
                new_id
            };

            let role = recovered_role.unwrap_or_else(|| "programmer".to_string());

            // Re-create system config
            let mut config_rows = Vec::new();
            config_rows.push(MemoryRow {
                vector: vec![0.0; EMBEDDING_DIMS as usize],
                text: role.clone(),
                tag: "system:role".to_string(),
                memory_type: "system_config".to_string(),
                file_path: "config".to_string(),
                line_start: 0,
                session_id: "global".to_string(),
                name: String::new(),
                references: "[]".to_string(),
                depends_on: "[]".to_string(),
                status: "active".to_string(),
                mtime_secs: 0,
                size_bytes: 0,
                created_at: now,
            });
            config_rows.push(MemoryRow {
                vector: vec![0.0; EMBEDDING_DIMS as usize],
                text: project_id,
                tag: "system:project_id".to_string(),
                memory_type: "system_config".to_string(),
                file_path: "config".to_string(),
                line_start: 0,
                session_id: "global".to_string(),
                name: String::new(),
                references: "[]".to_string(),
                depends_on: "[]".to_string(),
                status: "active".to_string(),
                mtime_secs: 0,
                size_bytes: 0,
                created_at: now,
            });
            let _ = table.delete("memory_type = 'system_config'").await;
            add_rows(&table, config_rows).await?;

            if preserve_episodic && !recovered_rows.is_empty() {
                println!("Re-importing {} episodic memories...", recovered_rows.len());
                let mut embedder = init_embedder(cache_dir.clone(), true)?;

                let mut docs = Vec::with_capacity(recovered_rows.len());
                for r in &recovered_rows {
                    let prefix = if r.memory_type == "trace" {
                        "trace"
                    } else if r.memory_type == "thought" {
                        "thought"
                    } else {
                        "passage"
                    };
                    docs.push(format!("{}: {}", prefix, r.text));
                }
                let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
                let embeddings = embedder.embed(refs, None)?;

                let mut rows_to_add = Vec::new();
                for (r, vector) in recovered_rows.into_iter().zip(embeddings.into_iter()) {
                    rows_to_add.push(MemoryRow {
                        vector,
                        text: r.text,
                        tag: r.tag.unwrap_or_else(|| "imported".to_string()),
                        memory_type: r.memory_type,
                        file_path: r.file_path,
                        line_start: r.line_start,
                        session_id: r.session_id,
                        name: r.name,
                        references: r.references,
                        depends_on: r.depends_on,
                        status: r.status,
                        mtime_secs: r.mtime_secs,
                        size_bytes: r.size_bytes,
                        created_at: r.created_at,
                    });
                }
                add_rows(&table, rows_to_add).await?;
            }

            if !skip_learn {
                do_learn(&project_root, &db_path, cache_dir.clone(), &root_dir, concurrency).await?;
            } else {
                println!("Skipped code re-index. Run `mem learn {}` when ready.", root_dir.display());
            }

            println!("Rebuild complete. Backup saved at {}", backup_path.display());
        }
        Command::Doctor => {
            println!("Mandrid Doctor");
            println!("- Project root: {}", project_root.display());
            println!("- DB path: {}", db_path.display());

            let version_path = db_path.join("FORMAT_VERSION");
            if version_path.exists() {
                let raw = fs::read_to_string(&version_path).unwrap_or_default();
                let found = raw.trim().parse::<u32>().unwrap_or(0);
                if found == DB_FORMAT_VERSION {
                    println!("- DB format: {} (ok)", found);
                } else {
                    println!(
                        "- DB format: {} (expected {}) -> run `mem rebuild`",
                        found, DB_FORMAT_VERSION
                    );
                }
            } else if db_path.exists() {
                println!("- DB format: unknown (no FORMAT_VERSION marker)");
            } else {
                println!("- DB format: n/a (no DB yet; run `mem init`)");
            }

            // List backups
            if let Some(parent) = db_path.parent() {
                if let Ok(entries) = fs::read_dir(parent) {
                    let mut backups = Vec::new();
                    for e in entries.flatten() {
                        let p = e.path();
                        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                            if name.starts_with(".mem_db.bak.") {
                                backups.push(p);
                            }
                        }
                    }
                    backups.sort();
                    if !backups.is_empty() {
                        println!("- Backups:");
                        for b in backups.iter().rev().take(3) {
                            println!("  - {}", b.display());
                        }
                    }
                }
            }

            // Attempt to open DB
            match open_table(&db_path).await {
                Ok(_) => println!("- DB open: ok"),
                Err(e) => {
                    println!("- DB open: failed ({})", e);
                    println!("  Suggested fix: `mem rebuild` (or `mem init` if brand new)");
                }
            }
        }
        Command::Digest { file_path } => {
            let mut embedder = init_embedder(cache_dir.clone(), true)?;
            let table = open_or_create_table(&db_path).await?;
            
            let abs_path = fs::canonicalize(&file_path).unwrap_or(file_path.clone());
            let rel_path = abs_path.strip_prefix(&project_root).unwrap_or(&abs_path);
            
            let rows = digest_file_logic(&mut embedder, &abs_path, rel_path).await?;
            add_rows(&table, rows).await?;
            println!("Digested {}", file_path.display());
        }
        Command::Learn { root_dir, concurrency } => {
            do_learn(&project_root, &db_path, cache_dir.clone(), &root_dir, concurrency).await?;
        }
        Command::Pack {
            input,
            session,
            scope,
            token_budget,
            k_episodic,
            k_code,
            include_code,
            include_reasoning,
            max_age_days,
            include_types,
            exclude_types,
            type_caps,
            rerank,
            vector_only,
            json,
        } => {
            let table = open_table(&db_path).await.with_context(|| {
                format!(
                    "Database not found at {}. Run `mem init` first.",
                    db_path.display()
                )
            })?;

            let session_id = session.unwrap_or_else(|| "default".to_string());

            // Role
            let role = {
                let stream = table
                    .query()
                    .only_if("tag = 'system:role'")
                    .limit(1)
                    .execute()
                    .await?;
                let batches: Vec<RecordBatch> = stream.try_collect().await?;
                let rows = decode_rows(&batches, None)?;
                rows.first()
                    .map(|r| r.text.clone())
                    .unwrap_or_else(|| "programmer".to_string())
            };

            // Active task title
            let active_task_title: Option<String> = {
                let filter = match scope {
                    ContextScope::Session => format!(
                        "memory_type = 'task' AND status = 'active' AND (session_id = '{}' OR session_id = 'global')",
                        escape_lancedb_str(&session_id)
                    ),
                    ContextScope::Project => "memory_type = 'task' AND status = 'active'".to_string(),
                };
                let stream = table.query().only_if(filter).limit(5).execute().await?;
                let batches: Vec<RecordBatch> = stream.try_collect().await?;
                let mut rows = decode_rows(&batches, None)?;
                rows.sort_by(|a, b| b.created_at.cmp(&a.created_at));
                rows.first()
                    .map(|t| t.text.lines().next().unwrap_or("").trim().to_string())
                    .filter(|s| !s.is_empty())
            };

            // Last failure summary
            let last_failure: Option<String> = {
                let filter = match scope {
                    ContextScope::Session => format!(
                        "memory_type = 'trace' AND status = 'failure' AND session_id = '{}'",
                        escape_lancedb_str(&session_id)
                    ),
                    ContextScope::Project => {
                        "memory_type = 'trace' AND status = 'failure'".to_string()
                    }
                };
                let stream = table.query().only_if(filter).limit(10).execute().await?;
                let batches: Vec<RecordBatch> = stream.try_collect().await?;
                let mut rows = decode_rows(&batches, None)?;
                rows.sort_by(|a, b| b.created_at.cmp(&a.created_at));
                rows.first().map(|f| {
                    let cmd = if f.name.is_empty() {
                        f.tag.clone().unwrap_or_else(|| "unknown".to_string())
                    } else {
                        f.name.clone()
                    };
                    let first = f.text.lines().next().unwrap_or("").trim();
                    if first.is_empty() {
                        cmd
                    } else {
                        format!("{} :: {}", cmd, truncate_chars(first, 180))
                    }
                })
            };

            let mut embedder = init_embedder(cache_dir.clone(), false)?;
            let query_embedding = embed_prefixed(&mut embedder, "query", &input)?;

            let include_types = normalize_memory_types(&include_types);
            let exclude_types = normalize_memory_types(&exclude_types);
            let type_caps = parse_type_caps(&type_caps);
            let mut filter_parts = vec![
                "memory_type != 'code'".to_string(),
                "memory_type != 'system_config'".to_string(),
            ];
            if !include_reasoning {
                filter_parts.push("memory_type != 'reasoning'".to_string());
            }
            if let Some(days) = max_age_days {
                let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
                let window = days.saturating_mul(86400);
                if window > 0 {
                    let threshold = now.saturating_sub(window);
                    filter_parts.push(format!("created_at >= {}", threshold));
                }
            }
            if !include_types.is_empty() {
                let list = include_types
                    .iter()
                    .map(|t| format!("'{}'", escape_lancedb_str(t)))
                    .collect::<Vec<_>>()
                    .join(", ");
                filter_parts.push(format!("memory_type IN ({})", list));
            }
            if !exclude_types.is_empty() {
                let list = exclude_types
                    .iter()
                    .map(|t| format!("'{}'", escape_lancedb_str(t)))
                    .collect::<Vec<_>>()
                    .join(", ");
                filter_parts.push(format!("memory_type NOT IN ({})", list));
            }
            match scope {
                ContextScope::Session => filter_parts.push(format!(
                    "session_id = '{}'",
                    escape_lancedb_str(&session_id)
                )),
                ContextScope::Project => {}
            }
            let episodic_filter = filter_parts.join(" AND ");

            let mut reranker = if rerank {
                Some(init_reranker(cache_dir.clone(), false)?)
            } else {
                None
            };

            let mut episodic_rows: Vec<DecodedRow> = if k_episodic == 0 {
                Vec::new()
            } else if let Some(ref mut rr) = reranker {
                ask_hybrid_with_reranker(
                    &table,
                    &query_embedding,
                    &input,
                    k_episodic,
                    vector_only,
                    Some(&episodic_filter),
                    rr,
                )
                .await?
            } else {
                ask_rrf_filtered(
                    &table,
                    &query_embedding,
                    &input,
                    k_episodic,
                    vector_only,
                    &episodic_filter,
                )
                .await?
            };

            let mut code_rows: Vec<DecodedRow> = if include_code && k_code > 0 {
                let code_filter = "memory_type = 'code'";
                if let Some(ref mut rr) = reranker {
                    ask_hybrid_with_reranker(
                        &table,
                        &query_embedding,
                        &input,
                        k_code,
                        vector_only,
                        Some(code_filter),
                        rr,
                    )
                    .await?
                } else {
                    ask_rrf_filtered(
                        &table,
                        &query_embedding,
                        &input,
                        k_code,
                        vector_only,
                        code_filter,
                    )
                    .await?
                }
            } else {
                Vec::new()
            };

            if !episodic_rows.is_empty() {
                let mut seen = HashSet::new();
                let mut deduped = Vec::with_capacity(episodic_rows.len());
                for row in episodic_rows.into_iter() {
                    let key = format!("{}:{}", row.memory_type, row.text);
                    if seen.insert(key) {
                        deduped.push(row);
                    }
                }
                episodic_rows = deduped;
            }

            if !type_caps.is_empty() && !episodic_rows.is_empty() {
                let mut counts: HashMap<String, usize> = HashMap::new();
                let mut capped = Vec::with_capacity(episodic_rows.len());
                for row in episodic_rows.into_iter() {
                    let key = row.memory_type.to_lowercase();
                    if let Some(limit) = type_caps.get(&key) {
                        let entry = counts.entry(key).or_insert(0);
                        if *entry >= *limit {
                            continue;
                        }
                        *entry += 1;
                    }
                    capped.push(row);
                }
                episodic_rows = capped;
            }

            if !code_rows.is_empty() {
                let mut seen = HashSet::new();
                let mut deduped = Vec::with_capacity(code_rows.len());
                for row in code_rows.into_iter() {
                    let key = format!("{}:{}:{}", row.file_path, row.line_start, row.text);
                    if seen.insert(key) {
                        deduped.push(row);
                    }
                }
                code_rows = deduped;
            }

            let pack = build_pack_text(
                &role,
                &session_id,
                &scope,
                token_budget,
                &input,
                active_task_title.as_deref(),
                last_failure.as_deref(),
                &episodic_rows,
                &code_rows,
            );

            if json {
                let payload = PackJsonPayload {
                    role: role.clone(),
                    session_id: session_id.clone(),
                    scope: scope_to_str(&scope).to_string(),
                    token_budget,
                    input: input.clone(),
                    active_task_title: active_task_title.clone(),
                    last_failure: last_failure.clone(),
                    episodic: episodic_rows
                        .iter()
                        .map(|r| to_pack_item(r, 700, false))
                        .collect(),
                    code: code_rows.iter().map(|r| to_pack_item(r, 900, true)).collect(),
                    pack,
                };
                println!("{}", serde_json::to_string_pretty(&payload)?);
            } else {
                print!("{}", pack);
            }
        }
        Command::Ask { question, json_output, limit, vector_only, scope, task_aware, rerank } => {
            if is_server_alive(3000).await && !task_aware {
                let client = reqwest::Client::new();
                let resp = client.get("http://localhost:3000/api/search")
                    .query(&[("q", &question), ("limit", &limit.to_string())])
                    .query(&[("rerank", &rerank.to_string()), ("vector_only", &vector_only.to_string())])
                    .send()
                    .await?;
                
                if resp.status().is_success() {
                    let results: Vec<DecodedRow> = resp.json().await?;
                    if json_output {
                        let payload = AskJsonPayload {
                            question,
                            results: results.into_iter().enumerate().map(|(i, r)| AskJsonResult {
                                rank: i + 1,
                                tag: r.tag,
                                text: r.text,
                                memory_type: r.memory_type,
                                file_path: Some(r.file_path),
                                line_start: Some(r.line_start),
                                created_at: r.created_at,
                            }).collect(),
                        };
                        println!("{}", serde_json::to_string_pretty(&payload)?);
                    } else {
                        for (i, r) in results.into_iter().enumerate() {
                            println!("\n{}. [{}] ({}:{})", i + 1, r.tag.unwrap_or_default(), r.file_path, r.line_start);
                            println!("{}", r.text);
                        }
                    }
                    return Ok(());
                }
            }

            let mut embedder = init_embedder(cache_dir.clone(), !json_output)?;
            let table = open_table(&db_path).await?;

            let mut final_query = question.clone();
            if task_aware {
                let task_stream = table.query().only_if("memory_type = 'task' AND status = 'active'").execute().await?;
                let task_batches: Vec<RecordBatch> = task_stream.try_collect().await?;
                let active_tasks = decode_rows(&task_batches, None)?;
                if let Some(task) = active_tasks.first() {
                    final_query = format!("Task Context: {} \nQuery: {}", task.text, question);
                }
            }

            let query_embedding = embed_prefixed(&mut embedder, "query", &final_query)?;
            
            let scope_filter = match scope {
                AskScope::All => None,
                AskScope::Code => Some("memory_type = 'code' AND memory_type != 'system_config'".to_string()),
                AskScope::Episodic => Some("memory_type != 'code' AND memory_type != 'system_config'".to_string()),
            };

            let results = if rerank {
                ask_hybrid(
                    &table,
                    &query_embedding,
                    &final_query,
                    limit,
                    vector_only,
                    cache_dir.clone(),
                    scope_filter.as_deref(),
                )
                .await?
            } else if let Some(f) = scope_filter.as_deref() {
                ask_rrf_filtered(&table, &query_embedding, &final_query, limit, vector_only, f).await?
            } else {
                ask_rrf(&table, &query_embedding, &final_query, limit, vector_only).await?
            };

            if json_output {
                let payload = AskJsonPayload {
                    question,
                    results: results.into_iter().enumerate().map(|(i, r)| AskJsonResult {
                        rank: i + 1,
                        tag: r.tag,
                        text: r.text,
                        memory_type: r.memory_type,
                        file_path: Some(r.file_path),
                        line_start: Some(r.line_start),
                        created_at: r.created_at,
                    }).collect(),
                };
                println!("{}", serde_json::to_string_pretty(&payload)?);
            } else {
                for (i, r) in results.into_iter().enumerate() {
                    println!("\n{}. [{}] ({}:{})", i + 1, r.tag.unwrap_or_default(), r.file_path, r.line_start);
                    println!("{}", r.text);
                }
            }
        }
        Command::Task { subcommand } => {
            let table = open_or_create_table(&db_path).await?;
            let mut embedder = init_embedder(cache_dir.clone(), true)?;
            let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
            match subcommand {
                TaskCommand::Create { name, description, depends_on } => {
                    let row = MemoryRow {
                        vector: embed_prefixed(&mut embedder, "task", &description)?,
                        text: description,
                        tag: format!("task:{}", name),
                        memory_type: "task".to_string(),
                        file_path: "task".to_string(),
                        line_start: 0,
                        session_id: "global".to_string(),
                        name: name.clone(),
                        references: "[]".to_string(),
                        depends_on: serde_json::to_string(&depends_on)?,
                        status: "pending".to_string(),
                        mtime_secs: 0,
                        size_bytes: 0,
                        created_at: now,
                    };
                    add_rows(&table, vec![row]).await?;
                    println!("Task created.");
                }
                TaskCommand::List => {
                    let stream = table.query().only_if("memory_type = 'task'").execute().await?;
                    let batches: Vec<RecordBatch> = stream.try_collect().await?;
                    let rows = decode_rows(&batches, None)?;
                    
                    println!("=== DEVELOPMENT TASKS ===");
                    for row in rows {
                        let name = row.tag.unwrap_or_default().replace("task:", "");
                        println!("\n[{}] {}", row.status.to_uppercase(), name);
                        println!("Description: {}", row.text);
                        if row.depends_on != "[]" {
                            println!("Depends on: {}", row.depends_on);
                        }
                    }
                }
                TaskCommand::Start { name } => {
                    let predicate = format!("tag = 'task:{}'", name.replace('\'', "''"));
                    table.update()
                        .only_if(&predicate)
                        .column("status", "'active'")
                        .execute()
                        .await?;
                    println!("Task '{}' started.", name);
                }
                TaskCommand::Finish { name } => {
                    let predicate = format!("tag = 'task:{}'", name.replace('\'', "''"));
                    table.update()
                        .only_if(&predicate)
                        .column("status", "'completed'")
                        .execute()
                        .await?;
                    println!("Task '{}' finished.", name);
                }
            }
        }
        Command::Debug { limit } => {
            let table = open_table(&db_path).await.with_context(|| {
                format!(
                    "Database/Table not found at {} (run `mem learn .` first)",
                    db_path.display()
                )
            })?;

            let count = table.count_rows(None).await?;
            println!("Total Memories: {count}");

            let stream = table.query().limit(limit).execute().await?;
            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            let rows = decode_rows(&batches, None)?;

            println!("\n___ RAW DATA INSPECTION ___");
            for (idx, row) in rows.into_iter().enumerate() {
                println!("\n--- Entry {} ---", idx + 1);
                println!("Tag: {}", row.tag.unwrap_or_default());
                println!("Type: {}", row.memory_type);
                println!("Session: {}", row.session_id);
                println!("File: {}:{}", row.file_path, row.line_start);
                println!("Timestamp: {}", row.created_at);
                println!("Content:");
                println!("{}", row.text);
                println!("{}", "-".repeat(30));
            }
        }
        Command::Map => {
            let table = open_table(&db_path).await?;
            let stream = table.query().only_if("memory_type = 'code'").execute().await?;
            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            let rows = decode_rows(&batches, None)?;
            let mut files: HashMap<String, Vec<String>> = HashMap::new();
            for row in rows {
                let symbol = row.text.lines().next().unwrap_or("").to_string();
                files.entry(row.file_path).or_default().push(symbol);
            }
            for (path, symbols) in files {
                println!("\n📁 {}", path);
                for s in symbols { println!("  - {}", s); }
            }
        }
        Command::Symbols { query, json } => {
            let table = open_table(&db_path).await?;
            let mut filter = "memory_type = 'code'".to_string();
            if let Some(ref q) = query {
                filter.push_str(&format!(" AND text LIKE '%{}%'", q.replace('\'', "''")));
            }

            let stream = table.query().only_if(filter).execute().await?;
            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            let rows = decode_rows(&batches, None)?;

            if json {
                let mut symbols = Vec::new();
                for row in rows {
                    if let Some(first_line) = row.text.lines().next() {
                        if first_line.starts_with("[Context:") {
                            let symbol = first_line.replace("[Context: ", "").replace(']', "");
                            symbols.push(serde_json::json!({
                                "symbol": symbol,
                                "file_path": row.file_path,
                                "line_start": row.line_start,
                                "name": row.name,
                            }));
                        }
                    }
                }
                println!("{}", serde_json::to_string_pretty(&symbols)?);
            } else {
                println!("=== DETERMINISTIC SYMBOL MAP ===");
                for row in rows {
                    if let Some(first_line) = row.text.lines().next() {
                        if first_line.starts_with("[Context:") {
                            let symbol = first_line.replace("[Context: ", "").replace(']', "");
                            println!("{} -> {}:{}", symbol, row.file_path, row.line_start);
                        }
                    }
                }
            }
        }
        Command::Impact { symbol, depth, json } => {
            let table = open_table(&db_path).await?;
            if !json {
                println!("🔍 Calculating Blast Radius for: {}", symbol);
            }
            
            let mut seen = std::collections::HashSet::new();
            let mut to_process = vec![(symbol.clone(), 0)];
            seen.insert(symbol.clone());

            let mut impact_map = Vec::new();

            while let Some((current_symbol, current_depth)) = to_process.pop() {
                if current_depth >= depth { continue; }

                let impacted = find_impacted(&table, &current_symbol).await?;
                if impacted.is_empty() { continue; }

                for row in impacted {
                    let name = if row.name.is_empty() { "anonymous".to_string() } else { row.name.clone() };
                    
                    if json {
                        impact_map.push(serde_json::json!({
                            "source_symbol": current_symbol,
                            "impacted_symbol": name,
                            "file_path": row.file_path,
                            "line_start": row.line_start,
                            "depth": current_depth + 1
                        }));
                    } else {
                        let indent = "  ".repeat(current_depth + 1);
                        println!("{}↳ {} ({}:{})", indent, name, row.file_path, row.line_start);
                    }
                    
                    if !seen.contains(&name) {
                        seen.insert(name.clone());
                        to_process.push((name, current_depth + 1));
                    }
                }
            }

            if json {
                println!("{}", serde_json::to_string_pretty(&impact_map)?);
            }
        }
        Command::SyncGit { commits } => {
            let table = open_or_create_table(&db_path).await?;
            let mut embedder = init_embedder(cache_dir.clone(), true)?;
            let output = tokio::process::Command::new("git")
                .args(["log", "-p", &format!("-n{}", commits)])
                .current_dir(&project_root)
                .output()
                .await?;
            let log = String::from_utf8_lossy(&output.stdout);
            
            for commit_block in log.split("commit ").skip(1) {
                let full_text = format!("commit {}", commit_block);
                let embedding = embed_prefixed(&mut embedder, "passage", &full_text)?;
                let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
                let row = MemoryRow {
                    vector: embedding,
                    text: full_text,
                    tag: get_origin_tag("git:history"),
                    memory_type: "git_reasoning".to_string(),
                    file_path: "git".to_string(),
                    line_start: 0,
                    session_id: "global".to_string(),
                    name: "commit".to_string(),
                    references: "[]".to_string(),
                    depends_on: "[]".to_string(),
                    status: "n/a".to_string(),
                    mtime_secs: 0,
                    size_bytes: 0,
                    created_at: now,
                };
                add_rows(&table, vec![row]).await?;
            }
            println!("Synced {} commits.", commits);
        }
        Command::Stop => {
            if is_server_alive(3000).await {
                let client = reqwest::Client::new();
                let _ = client.post("http://localhost:3000/api/stop").send().await;
                println!("🛑 Mandrid daemon stopped.");
            } else {
                println!("Mandrid daemon is not running.");
            }
        }
        Command::Serve { port, background } => {
            if background {
                let args: Vec<String> = std::env::args().filter(|a| a != "--background").collect();
                let _ = std::process::Command::new(&args[0])
                    .args(&args[1..])
                    .spawn()
                    .expect("Failed to spawn background daemon");
                println!("🚀 Mandrid daemon started in background on port {}", port);
                return Ok(());
            }

            let embedder = init_embedder(cache_dir.clone(), true)?;
            let state = ServeState {
                active_db_path: Arc::new(Mutex::new(db_path.clone())),
                cache_dir: cache_dir.clone(),
                embedder: Arc::new(Mutex::new(embedder)),
                reranker: Arc::new(Mutex::new(None)),
            };

            let app = Router::new()
                .route("/", get(|| async { Html(include_str!("dashboard.html")) }))
                .route("/api/status", get(move |State(st): State<ServeState>| async move {
                    let db_p = st.active_db_path.lock().await;
                    Json(StatusResponse {
                        status: "ok".to_string(),
                        version: env!("CARGO_PKG_VERSION").to_string(),
                        db_path: db_p.display().to_string(),
                    })
                }))
                .route(
                    "/api/projects",
                    get(|State(_state): State<ServeState>| async move {
                        let projects = get_registered_projects();
                        let mut results = Vec::new();
                        for p in projects {
                            let db_p = p.join(DEFAULT_DB_DIR);
                            let status = if db_p.exists() { "initialized" } else { "missing" };
                            results.push(serde_json::json!({
                                "name": p.file_name().unwrap_or_default().to_string_lossy(),
                                "path": p.to_string_lossy(),
                                "status": status
                            }));
                        }
                        Json(results)
                    }),
                )
                .route("/api/switch", post(|State(state): State<ServeState>, Query(q): Query<HashMap<String, String>>| async move {
                    if let Some(path_str) = q.get("path") {
                        let path = PathBuf::from(path_str);
                        if path.exists() {
                            let mut db_p = state.active_db_path.lock().await;
                            *db_p = path.join(DEFAULT_DB_DIR);
                            return Json(serde_json::json!({"status": "switched", "active": path_str}));
                        }
                    }
                    Json(serde_json::json!({"status": "error", "message": "invalid path"}))
                }))
                .route(
                    "/api/log",
                    post(|State(st): State<ServeState>, Json(req): Json<LogRequest>| async move {
                        let mut embedder = st.embedder.lock().await;
                        let db_p = st.active_db_path.lock().await;
                        let table = match open_or_create_table(&db_p).await {
                            Ok(t) => t,
                            Err(_) => return Json(serde_json::json!({"error": "db_error"})),
                        };

                        let prefix = if req.memory_type == "trace" { "trace" } else { "passage" };
                        let embedding = match embed_prefixed(&mut embedder, prefix, &req.text) {
                            Ok(v) => v,
                            Err(_) => return Json(serde_json::json!({"error": "embed_error"})),
                        };
                        let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs();

                        let default_path = if req.memory_type == "trace" { "terminal" } else { "interaction" };
                        let row = MemoryRow {
                            vector: embedding,
                            text: req.text,
                            tag: req.tag,
                            memory_type: req.memory_type,
                            file_path: req.file_path.unwrap_or_else(|| default_path.to_string()),
                            line_start: 0,
                            session_id: req.session.unwrap_or_else(|| "default".to_string()),
                            name: req.name.unwrap_or_default(),
                            references: "[]".to_string(),
                            depends_on: "[]".to_string(),
                            status: req.status,
                            mtime_secs: 0,
                            size_bytes: 0,
                            created_at: now,
                        };
                        let _ = add_rows(&table, vec![row]).await;
                        Json(serde_json::json!({"status": "logged"}))
                    }),
                )
                .route(
                    "/api/senses",
                    get(|State(st): State<ServeState>| async move {
                        let db_p = st.active_db_path.lock().await;
                        let table = match open_table(&db_p).await {
                            Ok(t) => t,
                            Err(_) => return Json(serde_json::json!({"dead_code": [], "bottlenecks": []})),
                        };

                        let stream = match table.query().only_if("memory_type = 'code'").execute().await {
                            Ok(s) => s,
                            Err(_) => return Json(serde_json::json!({"dead_code": [], "bottlenecks": []})),
                        };
                        let batches: Vec<RecordBatch> = match stream.try_collect().await {
                            Ok(b) => b,
                            Err(_) => return Json(serde_json::json!({"dead_code": [], "bottlenecks": []})),
                        };
                        let all_symbols = decode_rows(&batches, None).unwrap_or_default();

                        let mut dead_code = Vec::new();
                        let mut bottlenecks = Vec::new();

                        for symbol_row in &all_symbols {
                            if symbol_row.name.is_empty() || symbol_row.name == "anonymous" { continue; }
                            
                            // Check references
                            let impacted = find_impacted(&table, &symbol_row.name).await.unwrap_or_default();
                            
                            if impacted.is_empty() {
                                dead_code.push(serde_json::json!({
                                    "name": symbol_row.name,
                                    "file": symbol_row.file_path
                                }));
                            } else if impacted.len() > 5 {
                                bottlenecks.push(serde_json::json!({
                                    "name": symbol_row.name,
                                    "count": impacted.len()
                                }));
                            }
                        }

                        bottlenecks.sort_by_key(|b| std::cmp::Reverse(b["count"].as_u64().unwrap_or(0)));

                        Json(serde_json::json!({
                            "dead_code": dead_code.into_iter().take(10).collect::<Vec<_>>(),
                            "bottlenecks": bottlenecks.into_iter().take(10).collect::<Vec<_>>()
                        }))
                    }),
                )
                .route(
                    "/api/graph",
                    get(|State(st): State<ServeState>| async move {
                        let db_p = st.active_db_path.lock().await;
                        let table = match open_table(&db_p).await {
                            Ok(t) => t,
                            Err(_) => return Json(serde_json::json!({"nodes": [], "edges": []})),
                        };

                        let stream = match table.query().only_if("memory_type = 'code'").execute().await {
                            Ok(s) => s,
                            Err(_) => return Json(serde_json::json!({"nodes": [], "edges": []})),
                        };
                        let batches: Vec<RecordBatch> = match stream.try_collect().await {
                            Ok(b) => b,
                            Err(_) => return Json(serde_json::json!({"nodes": [], "edges": []})),
                        };
                        let rows = decode_rows(&batches, None).unwrap_or_default();

                        let mut nodes = Vec::new();
                        let mut edges = Vec::new();
                        let mut seen_symbols = std::collections::HashSet::new();

                        for row in &rows {
                            if row.name.is_empty() { continue; }
                            if !seen_symbols.contains(&row.name) {
                                nodes.push(serde_json::json!({
                                    "id": row.name,
                                    "label": row.name,
                                    "type": "symbol",
                                    "file": row.file_path
                                }));
                                seen_symbols.insert(row.name.clone());
                            }

                            if let Ok(refs) = serde_json::from_str::<Vec<String>>(&row.references) {
                                for r in refs {
                                    edges.push(serde_json::json!({
                                        "source": row.name,
                                        "target": r
                                    }));
                                }
                            }
                        }

                        Json(serde_json::json!({ "nodes": nodes, "edges": edges }))
                    }),
                )
                .route(
                    "/api/memories",
                    get(|State(st): State<ServeState>, Query(q): Query<MemoriesQuery>| async move {
                        let db_p = st.active_db_path.lock().await;
                        let table = match open_table(&db_p).await {
                            Ok(t) => t,
                            Err(_) => return Json(Vec::<DecodedRow>::new()),
                        };

                        let limit = q.limit.unwrap_or(500).min(5000);
                        let stream = match table.query().limit(limit).execute().await {
                            Ok(s) => s,
                            Err(_) => return Json(Vec::<DecodedRow>::new()),
                        };
                        let batches: Vec<RecordBatch> = match stream.try_collect().await {
                            Ok(b) => b,
                            Err(_) => return Json(Vec::<DecodedRow>::new()),
                        };
                        let mut rows = decode_rows(&batches, None).unwrap_or_default();
                        rows.sort_by_key(|r| std::cmp::Reverse(r.created_at));
                        Json(rows)
                    }),
                )
                .route(
                    "/api/search",
                    get(|State(st): State<ServeState>, Query(q): Query<SearchQuery>| async move {
                        let db_p = st.active_db_path.lock().await;
                        let table = match open_table(&db_p).await {
                            Ok(t) => t,
                            Err(_) => return Json(Vec::<DecodedRow>::new()),
                        };

                        let limit = q.limit.unwrap_or(25).min(200);
                        let rerank = q.rerank.unwrap_or(false);
                        let vector_only = q.vector_only.unwrap_or(false);

                        let query_text = q.q;
                        if query_text.trim().is_empty() {
                            return Json(Vec::<DecodedRow>::new());
                        }

                        let query_vec = {
                            let mut embedder = st.embedder.lock().await;
                            match embed_prefixed(&mut embedder, "query", &query_text) {
                                Ok(v) => v,
                                Err(_) => return Json(Vec::<DecodedRow>::new()),
                            }
                        };

                        let rows = if rerank {
                            let mut reranker_guard = st.reranker.lock().await;
                            if reranker_guard.is_none() {
                                match init_reranker(st.cache_dir.clone(), false) {
                                    Ok(r) => *reranker_guard = Some(r),
                                    Err(_) => return Json(Vec::<DecodedRow>::new()),
                                }
                            }
                            match reranker_guard.as_mut() {
                            Some(r) => ask_hybrid_with_reranker(&table, &query_vec, &query_text, limit, false, None, r)
                                    .await
                                    .unwrap_or_default(),
                                None => Vec::new(),
                            }
                        } else {
                            ask_rrf(&table, &query_vec, &query_text, limit, vector_only)
                                .await
                                .unwrap_or_default()
                        };


                        Json(rows)
                    }),
                )
                .route(
                    "/api/check-risk",
                    get(|State(st): State<ServeState>, Query(q): Query<SearchQuery>| async move {
                        let db_p = st.active_db_path.lock().await;
                        let table = match open_table(&db_p).await {
                            Ok(t) => t,
                            Err(_) => return Json(serde_json::json!({"risk": null})),
                        };

                        let mut embedder = st.embedder.lock().await;
                        let query_vec = match embed_prefixed(&mut embedder, "query", &q.q) {
                            Ok(v) => v,
                            Err(_) => return Json(serde_json::json!({"risk": null})),
                        };

                        let risk = check_risk(&table, &query_vec).await.unwrap_or(None);
                        Json(serde_json::json!({"risk": risk}))
                    }),
                )
                .route(
                    "/api/stop",
                    post(|| async {
                        std::process::exit(0);
                        #[allow(unreachable_code)]
                        Json(serde_json::json!({"status": "stopping"}))
                    }),
                )
                .layer(CorsLayer::permissive())
                .with_state(state);

            let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
            println!("🚀 Mandrid Dashboard at http://localhost:{}", port);
            axum::serve(listener, app).await?;
        }
        Command::Hook { shell } => {
            if shell == "zsh" {
                println!(r#"
# Mandrid Zsh Hook
# Source this in your .zshrc: source <(mem hook zsh)

 mandrid_preexec() {{
     export MANDRID_LAST_CMD="$1"
     export MANDRID_CMD_START=$(date +%s)
     # Risk check before running
     if [[ "$1" != "mem "* ]]; then
         mem check-risk "$1"
     fi
 }}


 mandrid_precmd() {{
     local exit_code=$?
     if [ -n "$MANDRID_LAST_CMD" ] && [[ "$MANDRID_LAST_CMD" != "mem log"* ]] && [[ "$MANDRID_LAST_CMD" != "mem hook"* ]]; then
         local now=$(date +%s)
         local duration=$((now - MANDRID_CMD_START))
         local status="success"
         if [ $exit_code -ne 0 ]; then status="failure"; fi
         # Log to Mandrid in background to avoid terminal lag
         (mem log --memory-type trace --status "$status" --tag terminal --session shell_history "Command: $MANDRID_LAST_CMD | Exit: $exit_code | Duration: ${{duration}}s" >/dev/null 2>&1 &)
         unset MANDRID_LAST_CMD
     fi
 }}

autoload -Uz add-zsh-hook
add-zsh-hook preexec mandrid_preexec
add-zsh-hook precmd mandrid_precmd
"#);
            } else if shell == "bash" {
                println!(r#"
# Mandrid Bash Hook
# Source this in your .bashrc: source <(mem hook bash)

 mandrid_log_cmd() {{
     local exit_code=$?
     local last_cmd=$(history 1 | sed 's/^[ ]*[0-9]*  //')
     # Filter out empty commands or the log command itself
     if [[ -n "$last_cmd" && "$last_cmd" != "mem log"* ]]; then
         local status="success"
         if [ $exit_code -ne 0 ]; then status="failure"; fi
         (mem log --memory-type trace --status "$status" --tag terminal --session shell_history "Command: $last_cmd | Exit: $exit_code" >/dev/null 2>&1 &)
     fi
 }}

PROMPT_COMMAND="mandrid_log_cmd; $PROMPT_COMMAND"
"#);
            } else if shell == "powershell" || shell == "pwsh" {
                println!(r#"
 # Mandrid PowerShell Hook
 # Add this to your $PROFILE: Invoke-Expression (& mem hook powershell)

 $global:MANDRID_LAST_HISTORY_ID = 0

 function prompt {{
     $last = Get-History -Count 1
     if ($last -and $last.Id -ne $global:MANDRID_LAST_HISTORY_ID) {{
         $global:MANDRID_LAST_HISTORY_ID = $last.Id

         $exit_code = 0
         if (-not $?) {{
             if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {{ $exit_code = $LASTEXITCODE }} else {{ $exit_code = 1 }}
         }}

         $status = if ($exit_code -eq 0) {{ "success" }} else {{ "failure" }}
         $text = "Command: $($last.CommandLine) | Exit: $exit_code"

         # Log to Mandrid in background
         Start-Process -FilePath "mem" -ArgumentList @(
             "log",
             "--memory-type", "trace",
             "--status", $status,
             "--tag", "terminal",
             "--session", "shell_history",
             $text
         ) -NoNewWindow
     }}
     "PS $($executionContext.SessionState.Path.CurrentLocation)> "
 }}
"#);
            } else {
                anyhow::bail!("Unsupported shell: {}", shell);
            }
        }
        Command::Auto { subcommand } => {
            match subcommand {
                AutoCommand::Init { force } => {
                    let auto_path = ensure_auto_dir(&project_root)?;
                    let config_path = auto_config_path(&project_root);
                    if config_path.exists() && !force {
                        println!("Auto config already exists at {}", config_path.display());
                    } else {
                        write_auto_config(&project_root, &AutoConfig::default())?;
                        println!("Wrote auto config to {}", config_path.display());
                    }

                    let gitignore_path = auto_gitignore_path(&project_root);
                    let mut ignore_content = if gitignore_path.exists() {
                        fs::read_to_string(&gitignore_path).unwrap_or_default()
                    } else {
                        String::new()
                    };
                    if !ignore_content.contains("queue.jsonl") {
                        if !ignore_content.ends_with('\n') && !ignore_content.is_empty() {
                            ignore_content.push('\n');
                        }
                        ignore_content.push_str("queue.jsonl\n");
                        fs::write(&gitignore_path, ignore_content)?;
                    }

                    println!("Auto memory initialized at {}", auto_path.display());
                }
                AutoCommand::Run {
                    event,
                    session,
                    cmd,
                    status,
                    duration_ms,
                    note,
                    commit,
                    paths,
                } => {
                    let _ = ensure_auto_dir(&project_root)?;
                    let config = load_auto_config(&project_root)?;
                    let session_id = session.unwrap_or_else(|| "default".to_string());

                    let entry = match event {
                        AutoEvent::GitCommit => {
                            build_git_commit_entry(&project_root, &config, &session_id, commit)
                                .await?
                        }
                        AutoEvent::Command => {
                            let cmd = cmd.ok_or_else(|| anyhow::anyhow!("--cmd is required for command events"))?;
                            let status = status.unwrap_or(0);
                            build_command_entry(
                                &project_root,
                                &config,
                                &session_id,
                                cmd,
                                status,
                                duration_ms,
                                note,
                            )?
                        }
                        AutoEvent::FileChange => {
                            build_file_change_entry(&project_root, &config, &session_id, paths)?
                        }
                    };

                    if let Some(entry) = entry {
                        if config.review.auto_approve {
                            store_auto_entries(
                                &db_path,
                                cache_dir.clone(),
                                &[entry],
                                Some(config.ttl.default_days),
                            )
                            .await?;
                            println!("Auto memory saved.");
                        } else {
                            append_auto_queue(&project_root, &entry)?;
                            println!("Auto memory queued. Use `mem auto approve` to persist.");
                        }
                    } else {
                        println!("No auto memory generated.");
                    }
                }
                AutoCommand::Status => {
                    let entries = load_auto_queue(&project_root)?;
                    println!("Pending auto memories: {}", entries.len());
                    for entry in entries.iter().take(5) {
                        println!("- {} | {}", entry.id, entry.summary);
                    }
                }
                AutoCommand::Approve { all, id } => {
                    let mut entries = load_auto_queue(&project_root)?;
                    if entries.is_empty() {
                        println!("No queued auto memories.");
                        return Ok(());
                    }
                    if !all && id.is_none() {
                        println!("Queued auto memories:");
                        for entry in &entries {
                            println!("- {} | {}", entry.id, entry.summary);
                        }
                        println!("Use --all or --id to approve.");
                        return Ok(());
                    }

                    let mut approved = Vec::new();
                    let mut remaining = Vec::new();
                    for entry in entries.drain(..) {
                        let matches = if all {
                            true
                        } else if let Some(ref target) = id {
                            entry.id == *target
                        } else {
                            false
                        };
                        if matches {
                            approved.push(entry);
                        } else {
                            remaining.push(entry);
                        }
                    }

                    if approved.is_empty() {
                        println!("No matching entries found.");
                        return Ok(());
                    }

                    let config = load_auto_config(&project_root)?;
                    store_auto_entries(
                        &db_path,
                        cache_dir.clone(),
                        &approved,
                        Some(config.ttl.default_days),
                    )
                    .await?;
                    write_auto_queue(&project_root, &remaining)?;
                    println!("Approved {} auto memories.", approved.len());
                }
                AutoCommand::Reject { all, id } => {
                    let mut entries = load_auto_queue(&project_root)?;
                    if entries.is_empty() {
                        println!("No queued auto memories.");
                        return Ok(());
                    }
                    if !all && id.is_none() {
                        println!("Queued auto memories:");
                        for entry in &entries {
                            println!("- {} | {}", entry.id, entry.summary);
                        }
                        println!("Use --all or --id to reject.");
                        return Ok(());
                    }

                    let mut remaining = Vec::new();
                    let mut rejected = 0usize;
                    for entry in entries.drain(..) {
                        let matches = if all {
                            true
                        } else if let Some(ref target) = id {
                            entry.id == *target
                        } else {
                            false
                        };
                        if matches {
                            rejected += 1;
                        } else {
                            remaining.push(entry);
                        }
                    }
                    write_auto_queue(&project_root, &remaining)?;
                    println!("Rejected {} auto memories.", rejected);
                }
                AutoCommand::Hook { action } => match action {
                    AutoHookAction::Install => install_auto_git_hook(&project_root)?,
                    AutoHookAction::Uninstall => uninstall_auto_git_hook(&project_root)?,
                },
            }
        }
        Command::Watch { root_dir } => {
            let (tx, rx) = channel();
            let mut watcher = RecommendedWatcher::new(tx, Config::default())?;
            watcher.watch(&root_dir, RecursiveMode::Recursive)?;
            let table = open_or_create_table(&db_path).await?;
            let mut embedder = init_embedder(cache_dir.clone(), false)?;
            let matcher = get_ignore_matcher(&project_root);
            for res in rx {
                if let Ok(event) = res {
                    for path in event.paths {
                        if !is_supported_file(&path) || is_ignored(&matcher, &path) { continue; }
                        
                        let rel_path_result = path.strip_prefix(&project_root).map(|p| p.to_path_buf());
                        let rel_path = match rel_path_result {
                            Ok(p) => p,
                            Err(_) => continue,
                        };
                        let tag = format!("code:{}", rel_path.display());

                        if event.kind.is_remove() || !path.exists() {
                            let _ = table.delete(format!("tag = '{}'", tag.replace('\'', "''")).as_str()).await;
                            eprintln!("Removed {}", rel_path.display());
                        } else if event.kind.is_modify() || event.kind.is_create() {
                            let abs_path = fs::canonicalize(&path).unwrap_or(path.clone());
                            if let Ok(rows) = digest_file_logic(&mut embedder, &abs_path, &rel_path).await {
                                let _ = table.delete(format!("tag = '{}'", tag.replace('\'', "''")).as_str()).await;
                                let _ = add_rows(&table, rows).await;
                                eprintln!("Updated {}", rel_path.display());
                            }
                        }
                    }
                }
            }
        }
        Command::Fix { session } => {
            let table = open_table(&db_path).await?;
            let session_id = session.unwrap_or_else(|| "default".to_string());
            
            // 1. Get the last failure
            let filter = format!("memory_type = 'trace' AND status = 'failure' AND session_id = '{}'", session_id.replace('\'', "''"));
            let stream = table.query().only_if(filter).limit(1).execute().await?;
            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            let failures = decode_rows(&batches, None)?;
            
            let last_failure = match failures.first() {
                Some(f) => f,
                None => {
                    println!("No recent failures found in session '{}'.", session_id);
                    return Ok(());
                }
            };

            println!("🔧 Analyzing last failure: {}", last_failure.name);
            println!("Error Sample: {}\n", last_failure.text.lines().take(5).collect::<Vec<_>>().join("\n"));

            // 2. Semantic search for relevant code context
            let mut embedder = init_embedder(cache_dir.clone(), false)?;
            // Use the error text as the query
            let query_vec = embed_prefixed(&mut embedder, "query", &last_failure.text)?;
            let context_rows = ask_rrf(&table, &query_vec, &last_failure.text, 3, false).await?;

            println!("--- SUGGESTED CONTEXT FOR REPAIR ---");
            for (i, row) in context_rows.iter().enumerate() {
                println!("{}. {}:{} ({})", i+1, row.file_path, row.line_start, row.memory_type);
            }

            println!("\n--- PROMPT FOR YOUR AGENT ---");
            println!("The following command failed:\n```\n{}\n```", last_failure.text);
            println!("\nRelevant code context detected by Mandrid:");
            for row in context_rows {
                println!("\nFile: {}\n```\n{}\n```", row.file_path, row.text);
            }
            println!("\nPlease analyze the error and propose a fix.");
        }
        Command::CheckRisk { command_text, session: _ } => {
            if is_server_alive(3000).await {
                let client = reqwest::Client::new();
                let resp = client.get("http://localhost:3000/api/check-risk")
                    .query(&[("q", &command_text)])
                    .send()
                    .await?;
                if resp.status().is_success() {
                    let body: serde_json::Value = resp.json().await?;
                    if let Some(risk) = body.get("risk").and_then(|v| v.as_str()) {
                        eprintln!("\n{}", risk);
                    }
                }
                return Ok(());
            }

            let table = open_table(&db_path).await?;
            let mut embedder = init_embedder(cache_dir.clone(), false)?;
            let query_vec = embed_prefixed(&mut embedder, "query", &command_text)?;
            
            if let Some(risk) = check_risk(&table, &query_vec).await? {
                eprintln!("\n{}", risk);
            }
        }
        Command::Why { symbol, session } => {
            let table = open_table(&db_path).await?;
            let _session_id = session.unwrap_or_else(|| "default".to_string());

            // 1. Find the symbol's last update
            let filter = format!("memory_type = 'code' AND name = '{}'", symbol.replace('\'', "''"));
            let stream = table.query().only_if(filter).limit(1).execute().await?;
            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            let symbols = decode_rows(&batches, None)?;

            let symbol_row = match symbols.first() {
                Some(s) => s,
                None => {
                    println!("Symbol '{}' not found in memory.", symbol);
                    return Ok(());
                }
            };

            println!("🕵️ Investigating symbol: {} ({}:{})", symbol, symbol_row.file_path, symbol_row.line_start);
            println!("Last Modified: {}\n", chrono::DateTime::from_timestamp(symbol_row.mtime_secs as i64, 0)
                .map(|d| d.to_rfc2822())
                .unwrap_or_else(|| "unknown".to_string()));

            // 2. Find reasoning traces/thoughts around that time
            // Look for thoughts within 1 hour of the mtime
            let window = 3600; // 1 hour
            let start = symbol_row.mtime_secs.saturating_sub(window);
            let end = symbol_row.mtime_secs.saturating_add(window);

            let reasoning_filter = format!(
                "memory_type IN ('thought', 'trace', 'git_reasoning', 'manual') AND created_at >= {} AND created_at <= {}",
                start, end
            );
            let r_stream = table.query().only_if(reasoning_filter).limit(5).execute().await?;
            let r_batches: Vec<RecordBatch> = r_stream.try_collect().await?;
            let reasons = decode_rows(&r_batches, None)?;

            if reasons.is_empty() {
                println!("No specific reasoning traces found within the modification window.");
                println!("Try searching globally with `mem ask \"why was {} changed?\"`", symbol);
            } else {
                println!("=== REASONING TRACES AT TIME OF CHANGE ===");
                for r in reasons {
                    println!("\n[{}] ({})", r.tag.unwrap_or_default(), r.memory_type);
                    // Print first few lines of reasoning
                    for line in r.text.lines().take(10) {
                        println!("  {}", line);
                    }
                    if r.text.lines().count() > 10 { println!("  ..."); }
                }
            }
        }
        Command::Sense { mode } => {
            let table = open_table(&db_path).await?;
            if mode == "dead-code" {
                println!("🕵️ Scanning for potential dead code (symbols with 0 incoming references)...");
                
                // Get all code symbols
                let stream = table.query().only_if("memory_type = 'code'").execute().await?;
                let batches: Vec<RecordBatch> = stream.try_collect().await?;
                let all_symbols = decode_rows(&batches, None)?;

                let mut dead_symbols = Vec::new();
                for symbol_row in &all_symbols {
                    if symbol_row.name.is_empty() || symbol_row.name == "anonymous" { continue; }
                    
                    // Check if anything references this name
                    let impacted = find_impacted(&table, &symbol_row.name).await?;
                    if impacted.is_empty() {
                        dead_symbols.push(symbol_row);
                    }
                }

                if dead_symbols.is_empty() {
                    println!("No dead symbols detected. Great job!");
                } else {
                    println!("=== POTENTIAL DEAD CODE ===");
                    for s in dead_symbols {
                        println!("- {} ({}:{})", s.name, s.file_path, s.line_start);
                    }
                }
            } else if mode == "bottlenecks" {
                println!("🕵️ Scanning for architectural bottlenecks (symbols referenced by many others)...");
                
                let stream = table.query().only_if("memory_type = 'code'").execute().await?;
                let batches: Vec<RecordBatch> = stream.try_collect().await?;
                let all_symbols = decode_rows(&batches, None)?;

                let mut counts = Vec::new();
                for symbol_row in &all_symbols {
                    if symbol_row.name.is_empty() { continue; }
                    let impacted = find_impacted(&table, &symbol_row.name).await?;
                    if impacted.len() > 5 {
                        counts.push((symbol_row.name.clone(), impacted.len(), symbol_row.file_path.clone()));
                    }
                }

                counts.sort_by_key(|c| std::cmp::Reverse(c.1));
                
                if counts.is_empty() {
                    println!("No significant bottlenecks detected.");
                } else {
                    println!("=== TOP BOTTLENECKS ===");
                    for (name, count, path) in counts.iter().take(10) {
                        println!("- {}: {} references ({})", name, count, path);
                    }
                }
            } else {
                println!("Unsupported sense mode: {}", mode);
            }
        }
        Command::Lsp => {
            let (connection, io_threads) = Connection::stdio();
            let server_capabilities = serde_json::to_value(&ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Options(TextDocumentSyncOptions {
                    change: Some(TextDocumentSyncKind::FULL),
                    ..Default::default()
                })),
                ..Default::default()
            })?;
            connection.initialize(server_capabilities)?;
            let table = open_or_create_table(&db_path).await?;
            let mut embedder = init_embedder(cache_dir.clone(), false)?;
            let matcher = get_ignore_matcher(&project_root);
            for msg in &connection.receiver {
                match msg {
                    Message::Notification(not) if not.method == DidChangeTextDocument::METHOD => {
                        let params: lsp_types::DidChangeTextDocumentParams = serde_json::from_value(not.params)?;
                        if let Ok(path) = params.text_document.uri.to_file_path() {
                            if !is_ignored(&matcher, &path) && is_supported_file(&path) {
                                if let Some(change) = params.content_changes.first() {
                                    let abs_path = fs::canonicalize(&path).unwrap_or(path.clone());
                                    let rel_path = abs_path.strip_prefix(&project_root).unwrap_or(&abs_path);
                                    if let Ok(rows) = digest_file_content_logic(&mut embedder, &change.text, &abs_path, rel_path).await {
                                        let tag = format!("code:{}", rel_path.display());
                                        let _ = table.delete(format!("tag = '{}'", tag.replace('\'', "''")).as_str()).await;
                                        let _ = add_rows(&table, rows).await;
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            io_threads.join()?;
        }
        Command::Tools { format } => {
            if format == "markdown" {
                println!(r#"# Mandrid AI Toolset
You are equipped with Mandrid, a local persistent memory layer. Use these tools via the `mem` CLI:

 ## Retrieval
 - `mem pack "<user message>"`: **ALWAYS-ON (runner)**. Tiny, token-budgeted context pack designed to inject every prompt.
 - `mem context [--human --compact --scope session] [--file <path>]`: **REQUIRED START**. Bootstrap your session with low-noise context (role, active tasks, recent failures).
 - `mem ask "<query>" [--rerank] [--scope code|episodic]`: Semantic/Hybrid search.
 - `mem impact <symbol>`: Deterministic blast-radius analysis. See who calls this function/struct.
 - `mem why <symbol>`: Attribution. See the reasoning traces captured when this symbol was last modified.

 ## Capture
 - `mem think "<reasoning>"`: **PROACTIVE**. Store your plan before acting.
 - `mem run -- <command>`: **CRITICAL**. Run terminal commands through this to record success/failure and output.
 - `mem capture "<summary>"`: Permanent record of changes and reasoning.
 - `mem auto <subcommand>`: Deterministic auto-memory from git/command events (LLM-free).

 ## Maintenance
 - `mem rebuild`: Rebuild the local DB after significant Mandrid updates (backs up `.mem_db`, regenerates it).
 - `mem doctor`: Diagnose DB health/version and show backups.
 - `mem prune`: Delete old memories by type and age.

## Data Model (practical)
- `memory_type=code`: indexed code chunks (AST-based)
- `memory_type=trace`: terminal command logs (success/failure)
- `memory_type=thought`: active plans/decisions
- `memory_type=task`: active goals
- `memory_type=reasoning`: captured diffs + rationale
- `memory_type=manual` / `interaction`: ad-hoc notes and agent logs

## Senses
- `mem sense dead-code`: Find unreferenced symbols.
- `mem sense bottlenecks`: Find over-referenced "God objects".
"#);
            } else {
                // Simplified JSON schema for agent capability negotiation
                let schema = serde_json::json!({
                     "tools": [
                        { "name": "mem pack", "description": "Build a tiny token-budgeted context pack for every prompt." },
                         { "name": "mem context", "description": "Retrieve project state, tasks, and recent failures." },
                         { "name": "mem ask", "description": "Search code and reasoning history." },
                         { "name": "mem impact", "description": "Graph-based dependency analysis." },
                          { "name": "mem think", "description": "Persist internal reasoning." },
                          { "name": "mem run", "description": "Execute terminal commands with telemetry capture." },
                          { "name": "mem auto", "description": "Deterministic auto-memory from git/command events." },
                          { "name": "mem rebuild", "description": "Rebuild the local DB after a Mandrid update." },
                          { "name": "mem doctor", "description": "Diagnose DB health/version and show backups." },
                          { "name": "mem prune", "description": "Delete old memories by type and age." }
                      ]
                  });
                println!("{}", serde_json::to_string_pretty(&schema)?);
            }
        }
        Command::Run { command, session, note } => {
            if command.is_empty() {
                anyhow::bail!("No command provided to run.");
            }

            let full_command = command.join(" ");
            
            // --- Pre-flight Risk Check ---
            let _ = tokio::process::Command::new("mem")
                .args(["check-risk", &full_command])
                .status()
                .await;
            // -----------------------------

            println!("🚀 Mandrid running: {}", full_command);

            let start_time = SystemTime::now();
            let output = tokio::process::Command::new(&command[0])
                .args(&command[1..])
                .current_dir(&project_root)
                .output()
                .await?;
            let end_time = SystemTime::now();
            let duration = end_time.duration_since(start_time).unwrap_or_default();

            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let status = output.status.code().unwrap_or(-1);

            // Print output to user
            if !stdout.is_empty() { print!("{}", stdout); }
            if !stderr.is_empty() { eprintln!("{}", stderr); }

            let (stdout_trimmed, stdout_truncated) = truncate_for_trace(&stdout, TRACE_OUTPUT_MAX_CHARS);
            let (stderr_trimmed, stderr_truncated) = truncate_for_trace(&stderr, TRACE_OUTPUT_MAX_CHARS);

            let mut embedder = init_embedder(cache_dir.clone(), false)?;
            let table = open_or_create_table(&db_path).await?;

            let mut reasoning = format!(
                "Command: {}\nStatus: {}\nDuration: {:?}\nNote: {}\n\nSTDOUT:\n{}\n\nSTDERR:\n{}",
                full_command,
                status,
                duration,
                note.unwrap_or_else(|| "Automated trace".to_string()),
                stdout_trimmed,
                stderr_trimmed
            );
            if stdout_truncated || stderr_truncated {
                reasoning.push_str("\n\n[output truncated for memory]");
            }

            let embedding = embed_prefixed(&mut embedder, "trace", &reasoning)?;
            let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
            
            let row = MemoryRow {
                vector: embedding,
                text: reasoning,
                tag: format!("run:{}", command[0]),
                memory_type: "trace".to_string(),
                file_path: "terminal".to_string(),
                line_start: 0,
                session_id: session.unwrap_or_else(|| "default".to_string()),
                name: command[0].clone(),
                references: "[]".to_string(),
                depends_on: "[]".to_string(),
                status: if status == 0 { "success".to_string() } else { "failure".to_string() },
                mtime_secs: 0,
                size_bytes: 0,
                created_at: now,
            };

            add_rows(&table, vec![row]).await?;
            println!("\n✅ Interaction recorded to Mandrid.");
        }
    }
    Ok(())
}

async fn do_learn(
    project_root: &Path,
    db_path: &Path,
    cache_dir: Option<PathBuf>,
    root_dir: &Path,
    concurrency: usize,
) -> Result<()> {
    let table = open_or_create_table(db_path).await?;
    let stream = table.query().only_if("memory_type = 'code'").execute().await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    let existing_rows = decode_rows(&batches, None)?;

    let mut db_file_map: HashMap<PathBuf, (u64, u64)> = HashMap::new();
    for row in existing_rows {
        db_file_map.insert(PathBuf::from(row.file_path), (row.mtime_secs, row.size_bytes));
    }

    let mut seen_on_disk = std::collections::HashSet::new();
    let matcher = get_ignore_matcher(project_root);
    let mut files_to_process = Vec::new();
    for result in WalkBuilder::new(root_dir)
        .hidden(false)
        .git_ignore(true)
        .build()
    {
        if let Ok(entry) = result {
            let path = entry.path();
            if path.is_file() && is_supported_file(path) && !is_ignored(&matcher, path) {
                let rel_path = path.strip_prefix(project_root).unwrap_or(path).to_path_buf();
                seen_on_disk.insert(rel_path.clone());

                let metadata = fs::metadata(path)?;
                let mtime = metadata
                    .modified()?
                    .duration_since(SystemTime::UNIX_EPOCH)?
                    .as_secs();
                let size = metadata.len();

                if let Some((db_mtime, db_size)) = db_file_map.get(&rel_path) {
                    if *db_mtime == mtime && *db_size == size {
                        continue;
                    }
                }
                files_to_process.push((path.to_path_buf(), rel_path));
            }
        }
    }

    // Prune files no longer on disk
    for db_rel_path in db_file_map.keys() {
        if !seen_on_disk.contains(db_rel_path) {
            let tag = format!("code:{}", db_rel_path.display());
            table
                .delete(format!("tag = '{}'", tag.replace('\'', "''")).as_str())
                .await?;
            println!("Pruned {}", db_rel_path.display());
        }
    }

    println!("Indexing {} files...", files_to_process.len());
    let mut processed = 0;
    for chunk in files_to_process.chunks(concurrency) {
        let mut tasks = Vec::new();
        for (abs_file, rel_file) in chunk {
            let abs_file = abs_file.clone();
            let rel_file = rel_file.clone();
            let c_dir = cache_dir.clone();
            tasks.push(tokio::spawn(async move {
                let mut embedder = init_embedder(c_dir, false)?;
                digest_file_logic(&mut embedder, &abs_file, &rel_file).await
            }));
        }
        for task in tasks {
            if let Ok(Ok(rows)) = task.await {
                if !rows.is_empty() {
                    let rel_path = &rows[0].file_path;
                    let tag = format!("code:{}", rel_path);
                    let _ = table
                        .delete(format!("tag = '{}'", tag.replace('\'', "''")).as_str())
                        .await;
                    add_rows(&table, rows).await?;
                    processed += 1;
                }
            }
        }
    }
    if processed > 0 {
        table
            .create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
            .execute()
            .await?;
    }
    println!("Finished: {} processed", processed);
    Ok(())
}

async fn digest_file_logic(embedder: &mut TextEmbedding, abs_path: &Path, rel_path: &Path) -> Result<Vec<MemoryRow>> {
    let content = tokio::fs::read_to_string(abs_path).await?;
    digest_file_content_logic(embedder, &content, abs_path, rel_path).await
}

async fn digest_file_content_logic(embedder: &mut TextEmbedding, content: &str, abs_path: &Path, rel_path: &Path) -> Result<Vec<MemoryRow>> {
    let chunks = structural_chunk(content, abs_path);
    let docs: Vec<String> = chunks.iter().map(|c| format!("passage: {}", c.text)).collect();
    let refs_to_embed: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
    let embeddings = embedder.embed(refs_to_embed, None)?;
    let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
    let metadata = tokio::fs::metadata(abs_path).await?;
    Ok(chunks.into_iter().zip(embeddings.into_iter()).map(|(chunk, vector)| MemoryRow {
        vector,
        text: chunk.text,
        tag: format!("code:{}", rel_path.display()),
        memory_type: "code".to_string(),
        file_path: rel_path.display().to_string(),
        line_start: chunk.line,
        session_id: "global".to_string(),
        name: chunk.name,
        references: serde_json::to_string(&chunk.references).unwrap_or_else(|_| "[]".to_string()),
        depends_on: "[]".to_string(),
        status: "n/a".to_string(),
        mtime_secs: metadata.modified().unwrap().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs(),
        size_bytes: metadata.len(),
        created_at: now
    }).collect())
}

// (ask logic lives in src/db.rs)
