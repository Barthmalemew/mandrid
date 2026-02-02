use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::index::Index;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::rerankers::rrf::RRFReranker;

use arrow_array::types::Float32Type;
use arrow_array::{Array, FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};

use tree_sitter::Parser as TSParser;
use ignore::WalkBuilder;

const DEFAULT_DB_PATH: &str = "./.mem_db";
const DEFAULT_TABLE_NAME: &str = "memories";
const EMBEDDING_DIMS: i32 = 384;

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

    /// Get a briefing of recent decisions and relevant project context.
    Brief {
        /// How many recent decisions to include
        #[arg(long, default_value_t = 5)]
        limit: usize,
        #[arg(long)]
        session: Option<String>,
    },

    /// Get optimized AI context for the current session.
    Context {
        #[arg(long)]
        session: Option<String>,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },

    /// Initialize Mandrid in the current directory (setup DB, gitignore, docs).
    Init,

    /// Ingest a single code file, chop it up, and save it.
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
    },

    /// Inspect the local LanceDB memory store.
    Debug {
        #[arg(long, default_value_t = 3)]
        limit: usize,
    },

    /// Show a high-level map of the codebase architecture.
    Map,
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

#[derive(Debug)]
struct MemoryRow {
    vector: Vec<f32>,
    text: String,
    tag: String,
    memory_type: String,
    file_path: String,
    line_start: u32,
    session_id: String,
    mtime_secs: u64,
    size_bytes: u64,
    created_at: u64,
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
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)?
                .as_secs();
            let row = MemoryRow {
                vector: embedding,
                text,
                tag,
                memory_type: "manual".to_string(),
                file_path: "manual".to_string(),
                line_start: 0,
                session_id: session.unwrap_or_else(|| "default".to_string()),
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
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)?
                .as_secs();
            let row = MemoryRow {
                vector: embedding,
                text,
                tag,
                memory_type: "interaction".to_string(),
                file_path: "interaction".to_string(),
                line_start: 0,
                session_id: session.unwrap_or_else(|| "default".to_string()),
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
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)?
                .as_secs();

            let row = MemoryRow {
                vector: embedding,
                text: full_text,
                tag: "capture".to_string(),
                memory_type: "reasoning".to_string(),
                file_path: "git".to_string(),
                line_start: 0,
                session_id: session.unwrap_or_else(|| "default".to_string()),
                mtime_secs: 0,
                size_bytes: 0,
                created_at: now,
            };
            add_rows(&table, vec![row]).await?;
            println!("Context captured");
        }
        Command::Brief { limit, session } => {
            let table = open_table(&db_path).await.with_context(|| {
                format!(
                    "Database/Table not found at {} (run `mem learn .` first)",
                    db_path.display()
                )
            })?;

            let session_filter = if let Some(s) = session {
                format!(" AND session_id = '{}'", s.replace('\'', "''"))
            } else {
                "".to_string()
            };

            // 1. Get recent reasoning/decisions (System 2)
            let reasoning_stream = table
                .query()
                .only_if(format!("memory_type = 'reasoning'{}", session_filter))
                .limit(limit)
                .execute()
                .await?;
            let reasoning_batches: Vec<RecordBatch> = reasoning_stream.try_collect().await?;
            let reasoning_rows = decode_rows(&reasoning_batches, None)?;

            // 2. Get top manual/knowledge facts (System 1)
            let knowledge_stream = table
                .query()
                .only_if(format!("(memory_type = 'manual' OR memory_type = 'interaction'){}", session_filter))
                .limit(limit)
                .execute()
                .await?;
            let knowledge_batches: Vec<RecordBatch> = knowledge_stream.try_collect().await?;
            let knowledge_rows = decode_rows(&knowledge_batches, None)?;

            println!("=== PROJECT BRIEFING ===");
            println!("\n[Recent Decisions & Reasoning]");
            if reasoning_rows.is_empty() {
                println!("No reasoning traces found.");
            }
            for (idx, row) in reasoning_rows.into_iter().enumerate() {
                println!("\n{}. {}", idx + 1, row.text);
            }

            println!("\n[Key Knowledge & Patterns]");
            if knowledge_rows.is_empty() {
                println!("No manual context found.");
            }
            for (idx, row) in knowledge_rows.into_iter().enumerate() {
                println!("\n{}. [{}] {}", idx + 1, row.tag.unwrap_or_default(), row.text);
            }
        }
        Command::Context { session, limit } => {
            let table = open_table(&db_path).await.with_context(|| {
                format!(
                    "Database/Table not found at {} (run `mem learn .` first)",
                    db_path.display()
                )
            })?;

            let session_id = session.unwrap_or_else(|| "default".to_string());
            let filter = format!("session_id = '{}' OR memory_type = 'manual'", session_id.replace('\'', "''"));

            let stream = table
                .query()
                .only_if(filter)
                .limit(limit)
                .execute()
                .await?;
            
            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            let rows = decode_rows(&batches, None)?;

            println!("<project_context>");
            println!("## Active Session: {}", session_id);
            for row in rows {
                println!("\n## {} ({}) [Session: {}]", row.tag.unwrap_or_else(|| "unknown".to_string()), row.memory_type, row.session_id);
                if row.memory_type == "code" {
                    println!("File: {}:{}", row.file_path, row.line_start);
                }
                println!("{}", row.text);
            }
            println!("</project_context>");
        }
        Command::Init => {
            // 1. Create DB Directory
            if !db_path.exists() {
                fs::create_dir_all(&db_path).context("Failed to create .mem_db")?;
                println!("Initialized empty memory at {}", db_path.display());
            } else {
                println!("Memory database already exists at {}", db_path.display());
            }

            // 2. Add to .gitignore
            let gitignore_path = Path::new(".gitignore");
            let mem_ignore = ".mem_db/";
            let mut current_ignore = if gitignore_path.exists() {
                fs::read_to_string(gitignore_path)?
            } else {
                String::new()
            };

            if !current_ignore.contains(mem_ignore) {
                if !current_ignore.is_empty() && !current_ignore.ends_with('\n') {
                    current_ignore.push('\n');
                }
                current_ignore.push_str(mem_ignore);
                current_ignore.push('\n');
                fs::write(gitignore_path, current_ignore)?;
                println!("Added .mem_db/ to .gitignore");
            }

            // 3. Create AGENTS.md
            let agents_md_path = Path::new("AGENTS.md");
            if !agents_md_path.exists() {
                let agents_content = r#"# Mandrid Memory Agent Instructions

This project uses **Mandrid** (`mem`) to store context and reasoning.
Before starting tasks, please verify the memory state.

## Quick Start
1. **Check Context:** Run `mem brief` to see recent decisions and reasoning.
2. **Search:** Run `mem ask --json "query"` to find relevant code or patterns.
3. **Capture:** When completing a significant task, run:
   `mem capture "Reasoning: why I did this change"`

## Tools
- `mem learn`: Re-indexes the codebase (run this if you suspect the index is stale).
- `mem capture`: Saves your git diff + reasoning to the permanent record.

Do not edit `.mem_db/` manually.
"#;
                fs::write(agents_md_path, agents_content)?;
                println!("Created AGENTS.md");
            }
        }
        Command::Digest { ref file_path } => {
            let mut embedder = init_embedder(cache_dir.clone(), true)?;
            let table = open_or_create_table(&db_path).await?;

            let rows = digest_file(&mut embedder, file_path).with_context(|| {
                format!("Failed to digest file {}", file_path.display())
            })?;
            add_rows(&table, rows).await?;
            println!("Digested {}", file_path.display());
        }
        Command::Learn { ref root_dir, concurrency } => {
            let table = open_or_create_table(&db_path).await?;
            
            // 1. Get existing file metadata from DB
            let stream = table.query().only_if("memory_type = 'code'").execute().await?;
            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            let existing_rows = decode_rows(&batches, None)?;
            
            let mut db_file_map: HashMap<PathBuf, (u64, u64)> = HashMap::new();
            for row in existing_rows {
                let path = PathBuf::from(row.file_path);
                db_file_map.insert(path, (row.mtime_secs, row.size_bytes));
            }

            // 2. Garbage Collection (Pruning)
            let mut pruned = 0usize;
            for path in db_file_map.keys() {
                if !path.exists() {
                    let tag = format!("code:{}", path.display());
                    let predicate = format!("tag = '{}'", tag.replace('\'', "''"));
                    let _ = table.delete(&predicate).await;
                    pruned += 1;
                    println!("Pruned missing file: {}", path.display());
                }
            }

            // 3. Indexing / Updating
            let mut files_to_process = Vec::new();
            for result in WalkBuilder::new(root_dir).build() {
                let entry = result?;
                let path = entry.path().to_path_buf();
                if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                    if is_supported_file(&path) {
                        let metadata = fs::metadata(&path)?;
                        let mtime = metadata.modified()?.duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
                        let size = metadata.len();
                        let abs_path = fs::canonicalize(&path).unwrap_or(path.clone());

                        if let Some((db_mtime, db_size)) = db_file_map.get(&abs_path) {
                            if *db_mtime == mtime && *db_size == size {
                                continue;
                            }
                        }
                        files_to_process.push(abs_path);
                    }
                }
            }

            println!("Found {} files to check/update", files_to_process.len());

            let mut processed = 0usize;
            let mut errors = 0usize;

            for chunk in files_to_process.chunks(concurrency) {
                let mut tasks = Vec::new();
                for file in chunk {
                    let file = file.clone();
                    let cache_dir = cache_dir.clone();
                    tasks.push(tokio::spawn(async move {
                        let mut embedder = init_embedder(cache_dir, false)?;
                        digest_file(&mut embedder, &file)
                    }));
                }

                for task in tasks {
                    match task.await? {
                        Ok(rows) => {
                            if !rows.is_empty() {
                                let abs_path = PathBuf::from(&rows[0].file_path);
                                let tag = format!("code:{}", abs_path.display());
                                let predicate = format!("tag = '{}'", tag.replace('\'', "''"));
                                let _ = table.delete(&predicate).await;
                                
                                add_rows(&table, rows).await?;
                                processed += 1;
                            }
                        }
                        Err(e) => {
                            eprintln!("Error: {}", e);
                            errors += 1;
                        }
                    }
                }
            }

            if processed > 0 || pruned > 0 {
                println!("Updating Full-Text Search index...");
                table
                    .create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
                    .execute()
                    .await?;
            }

            println!(
                "Finished: {} processed, {} pruned, {} errors",
                processed, pruned, errors
            );
        }
        Command::Ask {
            question,
            json_output,
            limit,
            vector_only,
        } => {
            let mut embedder = init_embedder(cache_dir.clone(), !json_output)?;
            let table = open_table(&db_path).await.with_context(|| {
                format!(
                    "Database/Table not found at {} (run `mem learn .` first)",
                    db_path.display()
                )
            })?;

            let query_embedding = embed_prefixed(&mut embedder, "query", &question)?;
            let results = ask_table(&table, &query_embedding, &question, limit, vector_only).await?;

            if json_output {
                let payload = AskJsonPayload {
                    question,
                    results,
                };
                println!("{}", serde_json::to_string(&payload)?);
            } else {
                println!("Results for: '{question}'");
                for result in results {
                    let tag = result.tag.unwrap_or_default();
                    let file_info = if let Some(path) = result.file_path {
                        format!(" ({}:{})", path, result.line_start.unwrap_or(0))
                    } else {
                        "".to_string()
                    };
                    println!("\n{}. Tag: {}{}", result.rank, tag, file_info);
                    println!("{}", result.text);
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
                let status = if contains_brackets(&row.text) {
                    "Brackets Found"
                } else {
                    "No Brackets"
                };

                println!("\n--- Entry {} ({}) ---", idx + 1, status);
                println!("Tag: {}", row.tag.unwrap_or_default());
                println!("Type: {}", row.memory_type);
                println!("File: {}:{}", row.file_path, row.line_start);
                println!("Timestamp: {}", row.created_at);
                println!("Content:");
                println!("{}", row.text);
                println!("{}", "-".repeat(30));
            }
        }
        Command::Map => {
            let table = open_table(&db_path).await.with_context(|| {
                format!(
                    "Database/Table not found at {} (run `mem learn .` first)",
                    db_path.display()
                )
            })?;

            let stream = table
                .query()
                .only_if("memory_type = 'code'")
                .execute()
                .await?;
            
            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            let rows = decode_rows(&batches, None)?;

            let mut files: HashMap<String, Vec<String>> = HashMap::new();
            for row in rows {
                // Extract symbol from context tag or first line
                let symbol = if let Some(first_line) = row.text.lines().next() {
                    if first_line.starts_with("[Context:") {
                        first_line.replace("[Context: ", "").replace(']', "")
                    } else {
                        // Fallback to a truncated version of the first line for non-structural chunks
                        first_line.chars().take(60).collect::<String>()
                    }
                } else {
                    "unknown".to_string()
                };
                files.entry(row.file_path).or_default().push(symbol);
            }

            println!("=== CODEBASE ARCHITECTURE MAP ===");
            let mut sorted_files: Vec<_> = files.into_iter().collect();
            sorted_files.sort_by(|a, b| a.0.cmp(&b.0));

            for (path, symbols) in sorted_files {
                let display_path = if let Ok(cwd) = std::env::current_dir() {
                    Path::new(&path).strip_prefix(cwd).unwrap_or(Path::new(&path)).display().to_string()
                } else {
                    path
                };
                println!("\n📁 {}", display_path);
                
                let unique_symbols: HashSet<_> = symbols.into_iter().collect();
                let mut sorted_symbols: Vec<_> = unique_symbols.into_iter().collect();
                sorted_symbols.sort();
                for sym in sorted_symbols {
                    if !sym.trim().is_empty() {
                        println!("  - {}", sym);
                    }
                }
            }
        }
    }

    Ok(())
}

fn init_embedder(cache_dir: Option<PathBuf>, show_progress: bool) -> Result<TextEmbedding> {
    let cache_dir = cache_dir
        .or_else(|| dirs::cache_dir().map(|p| p.join("mandrid")))
        .unwrap_or_else(|| PathBuf::from("./.mem_cache"));

    let options = InitOptions::new(EmbeddingModel::AllMiniLML6V2)
        .with_cache_dir(cache_dir)
        .with_show_download_progress(show_progress);

    Ok(TextEmbedding::try_new(options)?)
}

fn embed_prefixed(embedder: &mut TextEmbedding, prefix: &str, text: &str) -> Result<Vec<f32>> {
    let docs = vec![format!("{prefix}: {text}")];
    let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
    let mut embeddings = embedder.embed(refs, None)?;
    embeddings
        .pop()
        .context("Embedding generation returned no vectors")
}

fn memory_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                EMBEDDING_DIMS,
            ),
            true,
        ),
        Field::new("text", DataType::Utf8, false),
        Field::new("tag", DataType::Utf8, false),
        Field::new("memory_type", DataType::Utf8, false),
        Field::new("file_path", DataType::Utf8, false),
        Field::new("line_start", DataType::UInt32, false),
        Field::new("session_id", DataType::Utf8, false),
        Field::new("mtime_secs", DataType::UInt64, false),
        Field::new("size_bytes", DataType::UInt64, false),
        Field::new("created_at", DataType::UInt64, false),
    ]))
}

async fn open_or_create_table(db_path: &Path) -> Result<lancedb::Table> {
    let db = lancedb::connect(db_path.to_str().context("Invalid db path")?)
        .execute()
        .await?;

    match db.open_table(DEFAULT_TABLE_NAME).execute().await {
        Ok(table) => Ok(table),
        Err(_) => {
            let schema = memory_schema();
            let empty = empty_record_batch(schema.clone())?;
            let iter = vec![empty]
                .into_iter()
                .map(Ok::<_, ArrowError>);
            let batches = RecordBatchIterator::new(iter, schema);

            let table = db
                .create_table(DEFAULT_TABLE_NAME, Box::new(batches))
                .execute()
                .await?;
            
            // Create Full-Text Search index for hybrid precision
            table
                .create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
                .execute()
                .await?;

            Ok(table)
        }
    }
}

async fn open_table(db_path: &Path) -> Result<lancedb::Table> {
    let db = lancedb::connect(db_path.to_str().context("Invalid db path")?)
        .execute()
        .await?;
    Ok(db.open_table(DEFAULT_TABLE_NAME).execute().await?)
}

fn empty_record_batch(schema: SchemaRef) -> Result<RecordBatch> {
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        std::iter::empty::<Option<Vec<Option<f32>>>>(),
        EMBEDDING_DIMS,
    );
    let texts = StringArray::from_iter_values(std::iter::empty::<&str>());
    let tags = StringArray::from_iter_values(std::iter::empty::<&str>());
    let memory_types = StringArray::from_iter_values(std::iter::empty::<&str>());
    let file_paths = StringArray::from_iter_values(std::iter::empty::<&str>());
    let line_starts = arrow_array::UInt32Array::from_iter_values(std::iter::empty::<u32>());
    let session_ids = StringArray::from_iter_values(std::iter::empty::<&str>());
    let mtimes = arrow_array::UInt64Array::from_iter_values(std::iter::empty::<u64>());
    let sizes = arrow_array::UInt64Array::from_iter_values(std::iter::empty::<u64>());
    let created_ats = arrow_array::UInt64Array::from_iter_values(std::iter::empty::<u64>());

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(vectors),
            Arc::new(texts),
            Arc::new(tags),
            Arc::new(memory_types),
            Arc::new(file_paths),
            Arc::new(line_starts),
            Arc::new(session_ids),
            Arc::new(mtimes),
            Arc::new(sizes),
            Arc::new(created_ats),
        ],
    )?)
}

async fn add_rows(table: &lancedb::Table, rows: Vec<MemoryRow>) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }

    let schema = memory_schema();
    let batch = rows_to_batch(schema.clone(), &rows)?;
    let iter = vec![batch].into_iter().map(Ok::<_, ArrowError>);
    let batches = RecordBatchIterator::new(iter, schema);

    table.add(Box::new(batches)).execute().await?;
    Ok(())
}

fn rows_to_batch(schema: SchemaRef, rows: &[MemoryRow]) -> Result<RecordBatch> {
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        rows.iter().map(|row| {
            let values: Vec<Option<f32>> = row.vector.iter().copied().map(Some).collect();
            Some(values)
        }),
        EMBEDDING_DIMS,
    );

    let texts = StringArray::from_iter_values(rows.iter().map(|r| r.text.as_str()));
    let tags = StringArray::from_iter_values(rows.iter().map(|r| r.tag.as_str()));
    let memory_types = StringArray::from_iter_values(rows.iter().map(|r| r.memory_type.as_str()));
    let file_paths = StringArray::from_iter_values(rows.iter().map(|r| r.file_path.as_str()));
    let line_starts = arrow_array::UInt32Array::from_iter_values(rows.iter().map(|r| r.line_start));
    let session_ids = StringArray::from_iter_values(rows.iter().map(|r| r.session_id.as_str()));
    let mtimes = arrow_array::UInt64Array::from_iter_values(rows.iter().map(|r| r.mtime_secs));
    let sizes = arrow_array::UInt64Array::from_iter_values(rows.iter().map(|r| r.size_bytes));
    let created_ats = arrow_array::UInt64Array::from_iter_values(rows.iter().map(|r| r.created_at));

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(vectors),
            Arc::new(texts),
            Arc::new(tags),
            Arc::new(memory_types),
            Arc::new(file_paths),
            Arc::new(line_starts),
            Arc::new(session_ids),
            Arc::new(mtimes),
            Arc::new(sizes),
            Arc::new(created_ats),
        ],
    )?)
}

fn structural_chunk(content: &str, file_path: &Path) -> Vec<(String, u32)> {
    let extension = file_path.extension().and_then(|s| s.to_str()).unwrap_or("");
    let language = match extension {
        "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
        "js" | "jsx" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "py" => Some(tree_sitter_python::LANGUAGE.into()),
        _ => None,
    };

    let Some(lang) = language else {
        return chunk_text_line_aware(content, 500, 50);
    };

    let mut parser = TSParser::new();
    parser.set_language(&lang).expect("Error loading grammar");
    let tree = parser.parse(content, None).expect("Error parsing");
    
    let mut chunks = Vec::new();
    let mut cursor = tree.walk();
    let root = tree.root_node();

    // Iterate through top-level nodes
    for node in root.children(&mut cursor) {
        let kind = node.kind();
        let mut parent_context = String::new();

        // Basic structural context propagation
        if kind == "struct_item" || kind == "impl_item" || kind == "class_definition" || kind == "function_definition" {
            // For impl/class, we want to capture the name to prefix children if we recurse
            // For now, let's just label the top-level chunk
            let name = node.child_by_field_name("name")
                .or_else(|| node.child_by_field_name("declarator"))
                .map(|n| &content[n.start_byte()..n.end_byte()])
                .unwrap_or("unknown");
            parent_context = format!("{} {}", kind, name);
        }

        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        let text = &content[start_byte..end_byte];
        let start_line = node.start_position().row as u32 + 1;

        if text.len() > 1000 {
            for (sub_text, sub_line) in chunk_text_line_aware(text, 1000, 100) {
                let context_text = if !parent_context.is_empty() {
                    format!("[Context: {}]\n{}", parent_context, sub_text)
                } else {
                    sub_text
                };
                chunks.push((context_text, start_line + sub_line - 1));
            }
        } else if text.len() > 20 {
            let context_text = if !parent_context.is_empty() {
                format!("[Context: {}]\n{}", parent_context, text)
            } else {
                text.to_string()
            };
            chunks.push((context_text, start_line));
        }
    }

    if chunks.is_empty() && !content.trim().is_empty() {
        return chunk_text_line_aware(content, 500, 50);
    }

    chunks
}

fn chunk_text_line_aware(text: &str, chunk_size: usize, overlap: usize) -> Vec<(String, u32)> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut current_line = 0;

    while current_line < lines.len() {
        let mut current_chunk = String::new();
        let mut lines_in_chunk = 0;
        let start_line = current_line as u32 + 1;

        while current_line < lines.len() && current_chunk.len() < chunk_size {
            current_chunk.push_str(lines[current_line]);
            current_chunk.push('\n');
            current_line += 1;
            lines_in_chunk += 1;
        }

        if !current_chunk.trim().is_empty() {
            chunks.push((current_chunk, start_line));
        }

        if current_line < lines.len() {
            // Move back for overlap
            let back = lines_in_chunk.min(overlap / 50 + 1); // rough estimate
            current_line = current_line.saturating_sub(back).max(current_line - lines_in_chunk + 1);
        }
    }

    chunks
}

fn digest_file(embedder: &mut TextEmbedding, file_path: &Path) -> Result<Vec<MemoryRow>> {
    let abs_path = fs::canonicalize(file_path).unwrap_or(file_path.to_path_buf());
    let content = fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read {}", file_path.display()))?;

    let metadata = fs::metadata(file_path)?;
    let mtime_secs = metadata.modified()?.duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
    let size_bytes = metadata.len();

    let chunks = structural_chunk(&content, file_path);
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let docs: Vec<String> = chunks
        .iter()
        .map(|(chunk, _)| format!("passage: {chunk}"))
        .collect();
    let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
    let embeddings = embedder.embed(refs, None)?;

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();
    let mut rows = Vec::with_capacity(chunks.len());
    for ((chunk, line_start), vector) in chunks.into_iter().zip(embeddings.into_iter()) {
        rows.push(MemoryRow {
            vector,
            text: chunk,
            tag: format!("code:{}", abs_path.display()),
            memory_type: "code".to_string(),
            file_path: abs_path.display().to_string(),
            line_start,
            session_id: "global".to_string(),
            mtime_secs,
            size_bytes,
            created_at: now,
        });
    }

    Ok(rows)
}

fn is_supported_file(path: &Path) -> bool {
    let extensions: HashSet<&'static str> = [
        "py", "rs", "js", "ts", "jsx", "tsx", "md", "txt", "nix", "go", "c", "cpp", "h", "hpp",
    ]
    .into_iter()
    .collect();
    
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    extensions.contains(ext)
}

fn contains_brackets(text: &str) -> bool {
    text.contains('[') && text.contains(']')
}

async fn ask_table(
    table: &lancedb::Table,
    query_vector: &[f32],
    query_text: &str,
    limit: usize,
    vector_only: bool,
) -> Result<Vec<AskJsonResult>> {
    let query_builder = table
        .query()
        .nearest_to(query_vector)
        .context("Failed to build vector query")?;

    let stream = if vector_only {
        query_builder.limit(limit).execute().await?
    } else {
        query_builder
            .full_text_search(FullTextSearchQuery::new(query_text.to_owned()))
            .rerank(Arc::new(RRFReranker::default()))
            .limit(limit)
            .execute()
            .await?
    };

    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    let rows = decode_rows(&batches, Some(limit))?;

    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(idx, row)| AskJsonResult {
            rank: idx + 1,
            tag: row.tag,
            text: row.text,
            memory_type: row.memory_type,
            file_path: Some(row.file_path),
            line_start: Some(row.line_start),
            created_at: row.created_at,
        })
        .collect())
}

struct DecodedRow {
    tag: Option<String>,
    text: String,
    memory_type: String,
    file_path: String,
    line_start: u32,
    session_id: String,
    mtime_secs: u64,
    size_bytes: u64,
    created_at: u64,
}

fn decode_rows(
    batches: &[RecordBatch],
    limit: Option<usize>,
) -> Result<Vec<DecodedRow>> {
    let mut out = Vec::new();

    for batch in batches {
        let tag_col = batch.column_by_name("tag").context("Missing 'tag' column")?;
        let text_col = batch.column_by_name("text").context("Missing 'text' column")?;
        let type_col = batch.column_by_name("memory_type").context("Missing 'memory_type' column")?;
        let path_col = batch.column_by_name("file_path").context("Missing 'file_path' column")?;
        let line_col = batch.column_by_name("line_start").context("Missing 'line_start' column")?;
        let session_col = batch.column_by_name("session_id").context("Missing 'session_id' column")?;
        let mtime_col = batch.column_by_name("mtime_secs").context("Missing 'mtime_secs' column")?;
        let size_col = batch.column_by_name("size_bytes").context("Missing 'size_bytes' column")?;
        let time_col = batch.column_by_name("created_at").context("Missing 'created_at' column")?;

        let tags = tag_col.as_any().downcast_ref::<StringArray>().context("Expected 'tag' to be Utf8 StringArray")?;
        let texts = text_col.as_any().downcast_ref::<StringArray>().context("Expected 'text' to be Utf8 StringArray")?;
        let types = type_col.as_any().downcast_ref::<StringArray>().context("Expected 'memory_type' to be Utf8 StringArray")?;
        let paths = path_col.as_any().downcast_ref::<StringArray>().context("Expected 'file_path' to be Utf8 StringArray")?;
        let lines = line_col.as_any().downcast_ref::<arrow_array::UInt32Array>().context("Expected 'line_start' to be UInt32Array")?;
        let sessions = session_col.as_any().downcast_ref::<StringArray>().context("Expected 'session_id' to be Utf8 StringArray")?;
        let mtimes = mtime_col.as_any().downcast_ref::<arrow_array::UInt64Array>().context("Expected 'mtime_secs' to be UInt64Array")?;
        let sizes = size_col.as_any().downcast_ref::<arrow_array::UInt64Array>().context("Expected 'size_bytes' to be UInt64Array")?;
        let times = time_col.as_any().downcast_ref::<arrow_array::UInt64Array>().context("Expected 'created_at' to be UInt64Array")?;

        for row in 0..batch.num_rows() {
            let tag = if tags.is_null(row) { None } else { Some(tags.value(row).to_string()) };
            let text = if texts.is_null(row) { String::new() } else { texts.value(row).to_string() };
            let mem_type = if types.is_null(row) { "unknown".to_string() } else { types.value(row).to_string() };
            let path = if paths.is_null(row) { "unknown".to_string() } else { paths.value(row).to_string() };
            let line = lines.value(row);
            let session = if sessions.is_null(row) { "default".to_string() } else { sessions.value(row).to_string() };
            let mtime = mtimes.value(row);
            let size = sizes.value(row);
            let created_at = times.value(row);

            out.push(DecodedRow {
                tag,
                text,
                memory_type: mem_type,
                file_path: path,
                line_start: line,
                session_id: session,
                mtime_secs: mtime,
                size_bytes: size,
                created_at,
            });
            
            if let Some(limit) = limit {
                if out.len() >= limit {
                    return Ok(out);
                }
            }
        }
    }

    Ok(out)
}
