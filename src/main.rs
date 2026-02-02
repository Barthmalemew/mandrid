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
use fastembed::{TextEmbedding};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::index::Index;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::rerankers::rrf::RRFReranker;

use arrow_array::RecordBatch;
use ignore::WalkBuilder;
use notify::{Watcher, RecursiveMode, Config, RecommendedWatcher};
use std::sync::mpsc::channel;
use lsp_server::{Connection, Message};
use lsp_types::{
    ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind,
    TextDocumentSyncOptions, notification::{DidChangeTextDocument, Notification},
};

use crate::db::*;
use crate::chunker::*;
use crate::task::*;

const DEFAULT_DB_PATH: &str = "./.mem_db";

#[derive(Parser, Debug)]
#[command(name = "mem", version, about = "Mandrid (mem) - local persistent memory")]
struct Cli {
    #[arg(long, env = "MEM_DB_PATH", default_value = DEFAULT_DB_PATH)]
    db_path: PathBuf,

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
        /// Output in a more human-friendly format
        #[arg(long)]
        human: bool,
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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let db_path = cli.db_path.clone();
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
                depends_on: "[]".to_string(),
                status: "n/a".to_string(),
                mtime_secs: 0,
                size_bytes: 0,
                created_at: now,
            };
            add_rows(&table, vec![row]).await?;
            println!("Saved");
        }
        Command::Log { text, tag, session } => {
            let mut embedder = init_embedder(cache_dir.clone(), true)?;
            let table = open_or_create_table(&db_path).await?;

            let embedding = embed_prefixed(&mut embedder, "passage", &text)?;
            let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
            let row = MemoryRow {
                vector: embedding,
                text,
                tag,
                memory_type: "interaction".to_string(),
                file_path: "interaction".to_string(),
                line_start: 0,
                session_id: session.unwrap_or_else(|| "default".to_string()),
                depends_on: "[]".to_string(),
                status: "n/a".to_string(),
                mtime_secs: 0,
                size_bytes: 0,
                created_at: now,
            };
            add_rows(&table, vec![row]).await?;
            println!("Logged interaction");
        }
        Command::Capture { reasoning, session } => {
            let mut embedder = init_embedder(cache_dir.clone(), true)?;
            let table = open_or_create_table(&db_path).await?;

            let diff = std::process::Command::new("git")
                .args(["diff", "HEAD"])
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_else(|_| "No git diff available".to_string());

            let full_text = format!("Reasoning: {}\n\nChanges:\n{}", reasoning, diff);
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
                depends_on: "[]".to_string(),
                status: "n/a".to_string(),
                mtime_secs: 0,
                size_bytes: 0,
                created_at: now,
            };
            add_rows(&table, vec![row]).await?;
            println!("Context captured");
        }
        Command::Context { session, limit, human } => {
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

            let filter = format!("session_id = '{}' OR memory_type = 'manual' OR memory_type = 'task'", session_id.replace('\'', "''"));

            let stream = table.query().only_if(filter).limit(limit).execute().await?;
            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            let rows = decode_rows(&batches, None)?;

            if human {
                println!("=== MANDRID STATE SUMMARY ===");
                println!("Role: {}", role.to_uppercase());
                println!("Active Session: {}", session_id);
                if !active_tasks.is_empty() {
                    println!("\n[Active Tasks]");
                    for t in active_tasks {
                        println!("- {}: {}", t.tag.unwrap_or_default(), t.text);
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

                println!("\n## Active Session: {}", session_id);
                for row in rows {
                    if row.memory_type == "task" && row.status == "active" { continue; }
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
            let rows = digest_file_logic(&mut embedder, &file_path)?;
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

            let mut files_to_process = Vec::new();
            for result in WalkBuilder::new(&root_dir).build() {
                let entry = result?;
                let path = entry.path().to_path_buf();
                if entry.file_type().map(|t| t.is_file()).unwrap_or(false) && is_supported_file(&path) {
                    let metadata = fs::metadata(&path)?;
                    let mtime = metadata.modified()?.duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
                    let size = metadata.len();
                    let abs_path = fs::canonicalize(&path).unwrap_or(path.clone());

                    if let Some(&(db_mtime, db_size)) = db_file_map.get(&abs_path) {
                        if db_mtime == mtime && db_size == size { continue; }
                    }
                    files_to_process.push(abs_path);
                }
            }

            println!("Indexing {} files...", files_to_process.len());
            let mut processed = 0;
            for chunk in files_to_process.chunks(concurrency) {
                let mut tasks = Vec::new();
                for file in chunk {
                    let file = file.clone();
                    let c_dir = cache_dir.clone();
                    tasks.push(tokio::spawn(async move {
                        let mut embedder = init_embedder(c_dir, false)?;
                        digest_file_logic(&mut embedder, &file)
                    }));
                }
                for task in tasks {
                    if let Ok(Ok(rows)) = task.await {
                        if !rows.is_empty() {
                            let abs_path = PathBuf::from(&rows[0].file_path);
                            let tag = format!("code:{}", abs_path.display());
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
            let fetch_limit = if rerank { limit.max(10) * 2 } else { limit };
            let mut results = ask_table_logic(&table, &query_embedding, &final_query, fetch_limit, vector_only).await?;

            if rerank && !results.is_empty() {
                let mut reranker = init_reranker(cache_dir.clone(), !json_output)?;
                let documents: Vec<&str> = results.iter().map(|r| r.text.as_str()).collect();
                let reranked = reranker.rerank(final_query.as_str(), documents.as_slice(), false, None)?;
                
                let mut indexed_results: Vec<_> = results.into_iter().enumerate().collect();
                indexed_results.sort_by(|(idx_a, _), (idx_b, _)| {
                    let score_a = reranked.get(*idx_a).map(|r| r.score).unwrap_or(0.0);
                    let score_b = reranked.get(*idx_b).map(|r| r.score).unwrap_or(0.0);
                    score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
                });
                
                results = indexed_results.into_iter().map(|(_, r)| r).take(limit).collect();
            } else if rerank {
                results = results.into_iter().take(limit).collect();
            }


            if json_output {
                println!("{}", serde_json::to_string(&AskJsonPayload { question, results: results.into_iter().enumerate().map(|(i, r)| AskJsonResult { rank: i+1, tag: r.tag, text: r.text, memory_type: r.memory_type, file_path: Some(r.file_path), line_start: Some(r.line_start), created_at: r.created_at }).collect() })?);
            } else {
                for (i, r) in results.into_iter().enumerate() {
                    println!("\n{}. [{}] ({}:{})", i+1, r.tag.unwrap_or_default(), r.file_path, r.line_start);
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
        Command::Watch { root_dir } => {
            let (tx, rx) = channel();
            let mut watcher = RecommendedWatcher::new(tx, Config::default())?;
            watcher.watch(&root_dir, RecursiveMode::Recursive)?;
            let table = open_or_create_table(&db_path).await?;
            let mut embedder = init_embedder(cache_dir.clone(), false)?;
            for res in rx {
                if let Ok(event) = res {
                    for path in event.paths {
                        if is_supported_file(&path) {
                            if let Ok(rows) = digest_file_logic(&mut embedder, &path) {
                                let abs_path = fs::canonicalize(&path).unwrap_or(path.clone());
                                let _ = table.delete(format!("tag = 'code:{}'", abs_path.display().to_string().replace('\'', "''")).as_str()).await;
                                let _ = add_rows(&table, rows).await;
                                eprintln!("Updated {}", path.display());
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
            for msg in &connection.receiver {
                match msg {
                    Message::Notification(not) if not.method == DidChangeTextDocument::METHOD => {
                        let params: lsp_types::DidChangeTextDocumentParams = serde_json::from_value(not.params)?;
                        if let Ok(path) = params.text_document.uri.to_file_path() {
                            if let Some(change) = params.content_changes.first() {
                                if let Ok(rows) = digest_file_content_logic(&mut embedder, &change.text, &path) {
                                    let abs_path = fs::canonicalize(&path).unwrap_or(path.clone());
                                    let _ = table.delete(format!("tag = 'code:{}'", abs_path.display().to_string().replace('\'', "''")).as_str()).await;
                                    let _ = add_rows(&table, rows).await;
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
            let output = std::process::Command::new(&command[0])
                .args(&command[1..])
                .output()?;
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
fn digest_file_logic(embedder: &mut TextEmbedding, path: &Path) -> Result<Vec<MemoryRow>> {
    let content = fs::read_to_string(path)?;
    digest_file_content_logic(embedder, &content, path)
}

fn digest_file_content_logic(embedder: &mut TextEmbedding, content: &str, path: &Path) -> Result<Vec<MemoryRow>> {
    let chunks = structural_chunk(content, path);
    let docs: Vec<String> = chunks.iter().map(|(c, _)| format!("passage: {c}")).collect();
    let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
    let embeddings = embedder.embed(refs, None)?;
    let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
    let metadata = fs::metadata(path)?;
    Ok(chunks.into_iter().zip(embeddings.into_iter()).map(|((text, line_start), vector)| MemoryRow {
        vector, text, tag: format!("code:{}", path.display()), memory_type: "code".to_string(),
        file_path: path.display().to_string(), line_start, session_id: "global".to_string(),
        depends_on: "[]".to_string(), status: "n/a".to_string(),
        mtime_secs: metadata.modified().unwrap().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs(),
        size_bytes: metadata.len(), created_at: now
    }).collect())
}

async fn ask_table_logic(table: &lancedb::Table, query_vector: &[f32], query_text: &str, limit: usize, vector_only: bool) -> Result<Vec<DecodedRow>> {
    let query_builder = table.query().nearest_to(query_vector)?;
    let stream = if vector_only { query_builder.limit(limit).execute().await? } else {
        query_builder.full_text_search(FullTextSearchQuery::new(query_text.to_owned())).rerank(Arc::new(RRFReranker::default())).limit(limit).execute().await?
    };
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    decode_rows(&batches, Some(limit))
}
