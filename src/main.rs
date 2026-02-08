mod db;
mod chunker;
mod task;

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use fastembed::{TextEmbedding, TextRerank};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::index::Index;
use lancedb::index::scalar::FtsIndexBuilder;
use serde::Deserialize;

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
    routing::get,
    Json, Router,
};
use tower_http::cors::CorsLayer;

use tokio::sync::Mutex;

use crate::chunker::*;
use crate::db::*;
use crate::task::*;

const DEFAULT_DB_DIR: &str = ".mem_db";

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
    },

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

        /// Boost results relevant to the current active task
        #[arg(long)]
        task_aware: bool,

        /// Enable Cross-Encoder Reranking for hyper-precision
        #[arg(long)]
        rerank: bool,
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
    },

    /// Generate shell hooks for automated command capture.
    Hook {
        /// Shell type (zsh, bash, powershell)
        #[arg(default_value = "zsh")]
        shell: String,
    },

    /// Start a minimal LSP server to receive real-time code changes.
    Lsp,

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

#[derive(Clone)]
struct ServeState {
    db_path: PathBuf,
    cache_dir: Option<PathBuf>,
    embedder: Arc<Mutex<TextEmbedding>>,
    reranker: Arc<Mutex<Option<TextRerank>>>,
}

#[derive(Debug, Deserialize)]
struct MemoriesQuery {
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct SearchQuery {
    q: String,
    limit: Option<usize>,
    rerank: Option<bool>,
    vector_only: Option<bool>,
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
                tag,
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
            let mut embedder = init_embedder(cache_dir.clone(), true)?;
            let table = open_or_create_table(&db_path).await?;

            let prefix = if memory_type == "trace" { "trace" } else { "passage" };
            let embedding = embed_prefixed(&mut embedder, prefix, &text)?;
            let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();

            let default_path = if memory_type == "trace" { "terminal" } else { "interaction" };
            let row = MemoryRow {
                vector: embedding,
                text,
                tag,
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

            let mut filter = format!("created_at < {} AND memory_type IN ('trace', 'git_reasoning')", threshold);
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

                let summary_text = format!(
                    "SESSION SUMMARY [{}]: Compressed {} traces. Encountered {} failures. Final achievement: {}",
                    session_id,
                    total,
                    failures,
                    last_success.map(|r| r.text.lines().next().unwrap_or("n/a")).unwrap_or("None")
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
        Command::Capture { reasoning, session } => {
            let mut embedder = init_embedder(cache_dir.clone(), true)?;
            let table = open_or_create_table(&db_path).await?;

            let output = tokio::process::Command::new("git")
                .args(["diff", "HEAD"])
                .current_dir(&project_root)
                .output()
                .await
                .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_else(|_| "No git diff available".to_string());

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
            println!("Context captured");
        }
        Command::Context { session, limit, file, human, json } => {
            let table = open_table(&db_path).await.with_context(|| {
                format!("Database not found at {}. Run `mem init` first.", db_path.display())
            })?;

            let session_id = session.unwrap_or_else(|| "default".to_string());
            
            // 1. Get Project Role/Config
            let config_stream = table.query().only_if("memory_type = 'system_config'").execute().await?;
            let config_batches: Vec<RecordBatch> = config_stream.try_collect().await?;
            let config_rows = decode_rows(&config_batches, None)?;
            let role = config_rows.first().map(|r| r.text.as_str()).unwrap_or("programmer");

            // 2. Get Active Task(s)
            let task_stream = table.query().only_if("memory_type = 'task' AND status = 'active'").execute().await?;
            let task_batches: Vec<RecordBatch> = task_stream.try_collect().await?;
            let active_tasks = decode_rows(&task_batches, None)?;

            // 3. Get Recent Thoughts
            let thought_stream = table.query().only_if("memory_type = 'thought' AND status = 'active'").limit(5).execute().await?;
            let thought_batches: Vec<RecordBatch> = thought_stream.try_collect().await?;
            let active_thoughts = decode_rows(&thought_batches, None)?;

            // 4. Get Recent Failures (Negative Memory)
            let failure_stream = table.query().only_if("memory_type = 'trace' AND status = 'failure'").limit(3).execute().await?;
            let failure_batches: Vec<RecordBatch> = failure_stream.try_collect().await?;
            let recent_failures = decode_rows(&failure_batches, None)?;

            let mut filter = format!("(session_id = '{}' OR memory_type = 'manual' OR memory_type = 'task' OR memory_type = 'thought')", session_id.replace('\'', "''"));
            
            if let Some(f) = file {
                let abs_path = fs::canonicalize(&f).unwrap_or(f);
                let rel_path = abs_path.strip_prefix(&project_root).unwrap_or(&abs_path);
                filter.push_str(&format!(" AND (file_path = '{}' OR memory_type = 'manual' OR memory_type = 'task' OR memory_type = 'thought')", rel_path.display().to_string().replace('\'', "''")));
            }

            let stream = table.query().only_if(filter).limit(limit).execute().await?;
            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            let rows = decode_rows(&batches, None)?;

            if json {
                let payload = serde_json::json!({
                    "role": role,
                    "session_id": session_id,
                    "active_tasks": active_tasks,
                    "active_thoughts": active_thoughts,
                    "recent_failures": recent_failures,
                    "memories": rows
                });
                println!("{}", serde_json::to_string_pretty(&payload)?);
            } else if human {
                println!("=== MANDRID STATE SUMMARY ===");
                println!("Role: {}", role.to_uppercase());
                println!("Active Session: {}", session_id);
                if !active_tasks.is_empty() {
                    println!("\n[Active Tasks]");
                    for t in active_tasks {
                        println!("- {}: {}", t.tag.unwrap_or_default(), t.text);
                    }
                }
                if !active_thoughts.is_empty() {
                    println!("\n[🧠 BRAIN DUMP]");
                    for t in &active_thoughts {
                        println!("- {}", t.text);
                    }
                }
                if !recent_failures.is_empty() {
                    println!("\n[⚠️ RECENT FAILURES]");
                    for f in &recent_failures {
                        println!("- {}: {}", f.tag.as_ref().unwrap_or(&"unknown".to_string()), f.text.lines().next().unwrap_or(""));
                    }
                }
                println!("\n[Recent Memories]");
                for row in rows {
                    println!("- [{}] {}", row.tag.unwrap_or_default(), row.text.lines().next().unwrap_or(""));
                }
            } else {
                println!("<project_context>");
                println!("## AI Agent Role: {}", role.to_uppercase());
                if role == "assistant" {
                    println!("## CONSTRAINT: You are NOT allowed to write or modify code files directly.");
                }
                
                if !active_tasks.is_empty() {
                    println!("\n## CURRENT GOAL");
                    for task in &active_tasks {
                        println!("- **{}**: {}", task.tag.as_ref().unwrap_or(&"unknown".to_string()).replace("task:", ""), task.text);
                    }
                }

                if !active_thoughts.is_empty() {
                    println!("\n## 🧠 ACTIVE REASONING");
                    for t in &active_thoughts {
                        println!("{}", t.text);
                    }
                }

                if !recent_failures.is_empty() {
                    println!("\n## ⚠️ RECENT FAILURES (Negative Memory)");
                    for task in &recent_failures {
                        println!("- **{}**: {}", task.tag.as_ref().unwrap_or(&"unknown".to_string()), task.text.lines().next().unwrap_or(""));
                    }
                    println!("Avoid repeating the mistakes captured in these traces.");
                }

                println!("\n## Active Session: {}", session_id);
                for row in rows {
                    if (row.memory_type == "task" || row.memory_type == "thought") && row.status == "active" { continue; }
                    println!("\n## {} ({})", row.tag.unwrap_or_else(|| "unknown".to_string()), row.memory_type);
                    if row.memory_type == "code" {
                        println!("File: {}:{}", row.file_path, row.line_start);
                    }
                    println!("{}", row.text);
                }
                println!("</project_context>");
            }
        }
        Command::Init { role } => {
            if !db_path.exists() {
                fs::create_dir_all(&db_path).context("Failed to create .mem_db")?;
            }
            let table = open_or_create_table(&db_path).await?;
            let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
            let role_row = MemoryRow {
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
            };
            let _ = table.delete("memory_type = 'system_config'").await;
            add_rows(&table, vec![role_row]).await?;

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
                let agents_content = format!(r#"# Mandrid Memory Agent Instructions
Your role: **{}**
Run `mem context` to start.
"#, role);
                fs::write(agents_md_path, agents_content)?;
            }
            println!("Mandrid initialized with role: {}", role);
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
            let table = open_or_create_table(&db_path).await?;
            let stream = table.query().only_if("memory_type = 'code'").execute().await?;
            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            let existing_rows = decode_rows(&batches, None)?;
            
            let mut db_file_map: HashMap<PathBuf, (u64, u64)> = HashMap::new();
            for row in existing_rows {
                db_file_map.insert(PathBuf::from(row.file_path), (row.mtime_secs, row.size_bytes));
            }

            let mut seen_on_disk = std::collections::HashSet::new();
            let matcher = get_ignore_matcher(&project_root);
            let mut files_to_process = Vec::new();
            for result in WalkBuilder::new(&root_dir).hidden(false).git_ignore(true).build() {
                if let Ok(entry) = result {
                    let path = entry.path();
                    if path.is_file() && is_supported_file(path) && !is_ignored(&matcher, path) {
                        let rel_path = path.strip_prefix(&project_root).unwrap_or(path).to_path_buf();
                        seen_on_disk.insert(rel_path.clone());
                        
                        let metadata = fs::metadata(path)?;
                        let mtime = metadata.modified()?.duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
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
                    table.delete(format!("tag = '{}'", tag.replace('\'', "''")).as_str()).await?;
                    println!("Pruned {}", db_rel_path.display());
                }
            }


            let matcher = get_ignore_matcher(&project_root);
            let mut files_to_process = Vec::new();
            for result in WalkBuilder::new(&root_dir).hidden(false).git_ignore(true).build() {
                let entry = result?;
                let path = entry.path().to_path_buf();
                if entry.file_type().map(|t| t.is_file()).unwrap_or(false) && is_supported_file(&path) {
                    if is_ignored(&matcher, &path) {
                        continue;
                    }
                    let metadata = tokio::fs::metadata(&path).await?;
                    let mtime = metadata.modified()?.duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
                    let size = metadata.len();
                    
                    let abs_path = fs::canonicalize(&path).unwrap_or(path.clone());
                    let rel_path = abs_path.strip_prefix(&project_root).unwrap_or(&abs_path).to_path_buf();

                    if let Some(&(db_mtime, db_size)) = db_file_map.get(&rel_path) {
                        if db_mtime == mtime && db_size == size { continue; }
                    }
                    files_to_process.push((abs_path, rel_path));
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
                            let _ = table.delete(format!("tag = '{}'", tag.replace('\'', "''")).as_str()).await;
                            add_rows(&table, rows).await?;
                            processed += 1;
                        }
                    }
                }
            }
            if processed > 0 {
                table.create_index(&["text"], Index::FTS(FtsIndexBuilder::default())).execute().await?;
            }
            println!("Finished: {} processed", processed);
        }
        Command::Ask { question, json_output, limit, vector_only, task_aware, rerank } => {
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
            
            let results = if rerank {
                ask_hybrid(&table, &query_embedding, &final_query, limit, cache_dir.clone()).await?
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
                    tag: "git:history".to_string(),
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
        Command::Serve { port } => {
            let embedder = init_embedder(cache_dir.clone(), true)?;
            let state = ServeState {
                db_path: db_path.clone(),
                cache_dir: cache_dir.clone(),
                embedder: Arc::new(Mutex::new(embedder)),
                reranker: Arc::new(Mutex::new(None)),
            };

            let app = Router::new()
                .route("/", get(|| async { Html(include_str!("dashboard.html")) }))
                .route(
                    "/api/memories",
                    get(|State(state): State<ServeState>, Query(q): Query<MemoriesQuery>| async move {
                        let table = match open_table(&state.db_path).await {
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
                    get(|State(state): State<ServeState>, Query(q): Query<SearchQuery>| async move {
                        let table = match open_table(&state.db_path).await {
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
                            let mut embedder = state.embedder.lock().await;
                            match embed_prefixed(&mut embedder, "query", &query_text) {
                                Ok(v) => v,
                                Err(_) => return Json(Vec::<DecodedRow>::new()),
                            }
                        };

                        let rows = if rerank {
                            let mut reranker_guard = state.reranker.lock().await;
                            if reranker_guard.is_none() {
                                match init_reranker(state.cache_dir.clone(), false) {
                                    Ok(r) => *reranker_guard = Some(r),
                                    Err(_) => return Json(Vec::<DecodedRow>::new()),
                                }
                            }
                            match reranker_guard.as_mut() {
                                Some(r) => ask_hybrid_with_reranker(&table, &query_vec, &query_text, limit, r)
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
                        } else if event.kind.is_modify() || event.kind.is_create() || event.kind.is_access() {
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
        Command::Run { command, session, note } => {
            if command.is_empty() {
                anyhow::bail!("No command provided to run.");
            }

            let full_command = command.join(" ");
            println!("🚀 Mandrid running: {}", full_command);

            let start_time = SystemTime::now();
            let output = tokio::process::Command::new(&command[0])
                .args(&command[1..])
                .current_dir(&project_root)
                .output()
                .await?;
            let end_time = SystemTime::now();
            let duration = end_time.duration_since(start_time).unwrap_or_default();

            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let status = output.status.code().unwrap_or(-1);

            // Print output to user
            if !stdout.is_empty() { print!("{}", stdout); }
            if !stderr.is_empty() { eprintln!("{}", stderr); }

            let mut embedder = init_embedder(cache_dir.clone(), false)?;
            let table = open_or_create_table(&db_path).await?;

            let reasoning = format!(
                "Command: {}\nStatus: {}\nDuration: {:?}\nNote: {}\n\nSTDOUT:\n{}\n\nSTDERR:\n{}",
                full_command,
                status,
                duration,
                note.unwrap_or_else(|| "Automated trace".to_string()),
                stdout,
                stderr
            );

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

// Logic wrappers to bridge modules
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
