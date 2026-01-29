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

use arrow_array::types::Float32Type;
use arrow_array::{Array, FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};

const DEFAULT_DB_PATH: &str = "./.mem_db";
const DEFAULT_TABLE_NAME: &str = "memories";
const STATE_FILE_NAME: &str = "index_state.json";
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
    },

    /// Record an agent interaction (reasoning trace/decision).
    Log {
        /// What was done or decided
        text: String,
        /// Optional context tag (e.g. "auth-refactor")
        #[arg(long, default_value = "interaction")]
        tag: String,
    },

    /// Capture current context (git diff + reasoning) into memory.
    Capture {
        /// The reasoning or explanation for the current state/changes
        reasoning: String,
    },

    /// Get a briefing of recent decisions and relevant project context.
    Brief {
        /// How many recent decisions to include
        #[arg(long, default_value_t = 5)]
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
    },

    /// Find relevant memories or code chunks.
    Ask {
        question: String,

        #[arg(long = "json")]
        json_output: bool,

        #[arg(long, default_value_t = 3)]
        limit: usize,
    },

    /// Inspect the local LanceDB memory store.
    Debug {
        #[arg(long, default_value_t = 3)]
        limit: usize,
    },
}

#[derive(Debug, serde::Serialize)]
struct AskJsonResult {
    rank: usize,
    tag: Option<String>,
    text: String,
    memory_type: String,
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
    created_at: u64,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct FileMetadata {
    mtime_secs: u64,
    size_bytes: u64,
    indexed_at: u64,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Default)]
struct IndexState {
    files: HashMap<PathBuf, FileMetadata>,
}

impl IndexState {
    fn load(db_path: &Path) -> Result<Self> {
        let state_path = db_path.join(STATE_FILE_NAME);
        if !state_path.exists() {
            return Ok(Self::default());
        }
        let content = fs::read_to_string(&state_path)?;
        Ok(serde_json::from_str(&content).unwrap_or_default())
    }

    fn save(&self, db_path: &Path) -> Result<()> {
        let state_path = db_path.join(STATE_FILE_NAME);
        if let Some(parent) = state_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)?;
        fs::write(state_path, content)?;
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let db_path = cli.db_path.clone();
    let cache_dir = cli.cache_dir.clone();

    match cli.command {
        Command::Save { text, tag } => {
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
                created_at: now,
            };
            add_rows(&table, vec![row]).await?;
            println!("Saved");
        }
        Command::Log { text, tag } => {
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
                created_at: now,
            };
            add_rows(&table, vec![row]).await?;
            println!("Logged interaction");
        }
        Command::Capture { reasoning } => {
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
                created_at: now,
            };
            add_rows(&table, vec![row]).await?;
            println!("Context captured");
        }
        Command::Brief { limit } => {
            let table = open_table(&db_path).await.with_context(|| {
                format!(
                    "Database/Table not found at {} (run `mem learn .` first)",
                    db_path.display()
                )
            })?;

            // 1. Get recent reasoning/decisions (System 2)
            let reasoning_stream = table
                .query()
                .only_if("memory_type = 'reasoning'")
                .limit(limit)
                .execute()
                .await?;
            let reasoning_batches: Vec<RecordBatch> = reasoning_stream.try_collect().await?;
            let reasoning_rows = decode_rows(&reasoning_batches, None)?;

            // 2. Get top manual/knowledge facts (System 1)
            let knowledge_stream = table
                .query()
                .only_if("memory_type = 'manual' OR memory_type = 'interaction'")
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
            for (idx, (_tag, text, _type, _ts)) in reasoning_rows.into_iter().enumerate() {
                println!("\n{}. {}", idx + 1, text);
            }

            println!("\n[Key Knowledge & Patterns]");
            if knowledge_rows.is_empty() {
                println!("No manual context found.");
            }
            for (idx, (tag, text, _type, _ts)) in knowledge_rows.into_iter().enumerate() {
                println!("\n{}. [{}] {}", idx + 1, tag.unwrap_or_default(), text);
            }
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
        Command::Learn { ref root_dir } => {
            let mut embedder = init_embedder(cache_dir.clone(), true)?;
            let table = open_or_create_table(&db_path).await?;
            let mut state = IndexState::load(&db_path)?;

            // 1. Garbage Collection (Pruning)
            let mut pruned = 0usize;
            let tracked_files: Vec<PathBuf> = state.files.keys().cloned().collect();
            for path in tracked_files {
                if !path.exists() {
                    // File deleted from disk, remove from DB and State
                    let tag = format!("code:{}", path.display());
                    let predicate = format!("tag = '{}'", tag.replace('\'', "''"));
                    let _ = table.delete(&predicate).await;
                    
                    state.files.remove(&path);
                    pruned += 1;
                    println!("Pruned missing file: {}", path.display());
                }
            }

            // 2. Indexing / Updating
            let mut processed = 0usize;
            let mut skipped = 0usize;
            let mut errors = 0usize;

            for file in walk_repo_files(root_dir)? {
                match process_file_if_changed(&mut embedder, &table, &mut state, &file).await {
                    Ok(true) => processed += 1,
                    Ok(false) => skipped += 1,
                    Err(e) => {
                        eprintln!("Error processing {}: {}", file.display(), e);
                        errors += 1;
                    }
                }
            }
            state.save(&db_path)?;

            println!(
                "Finished: {} processed, {} skipped, {} pruned, {} errors",
                processed, skipped, pruned, errors
            );
        }
        Command::Ask {
            question,
            json_output,
            limit,
        } => {
            let mut embedder = init_embedder(cache_dir.clone(), !json_output)?;
            let table = open_table(&db_path).await.with_context(|| {
                format!(
                    "Database/Table not found at {} (run `mem learn .` first)",
                    db_path.display()
                )
            })?;

            let query_embedding = embed_prefixed(&mut embedder, "query", &question)?;
            let results = ask_table(&table, &query_embedding, limit).await?;

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
                    println!("\n{}. Tag: {}", result.rank, tag);
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
            for (idx, (tag, text, mem_type, created_at)) in rows.into_iter().enumerate() {
                let status = if contains_brackets(&text) {
                    "Brackets Found"
                } else {
                    "No Brackets"
                };

                println!("\n--- Entry {} ({}) ---", idx + 1, status);
                println!("Tag: {}", tag.unwrap_or_default());
                println!("Type: {}", mem_type);
                println!("Timestamp: {}", created_at);
                println!("Content:");
                println!("{text}");
                println!("{}", "-".repeat(30));
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
    let created_ats = arrow_array::UInt64Array::from_iter_values(std::iter::empty::<u64>());

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(vectors),
            Arc::new(texts),
            Arc::new(tags),
            Arc::new(memory_types),
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
    let created_ats = arrow_array::UInt64Array::from_iter_values(rows.iter().map(|r| r.created_at));

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(vectors),
            Arc::new(texts),
            Arc::new(tags),
            Arc::new(memory_types),
            Arc::new(created_ats),
        ],
    )?)
}

async fn process_file_if_changed(
    embedder: &mut TextEmbedding,
    table: &lancedb::Table,
    state: &mut IndexState,
    file_path: &Path,
) -> Result<bool> {
    let metadata = fs::metadata(file_path)?;
    let mtime = metadata
        .modified()?
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();
    let size = metadata.len();
    let abs_path = fs::canonicalize(file_path)?;

    // Check if we need to re-index
    if let Some(existing) = state.files.get(&abs_path) {
        if existing.mtime_secs == mtime && existing.size_bytes == size {
            return Ok(false);
        }
    }

    // Delete old memories for this file (using tag convention)
    let tag = format!("code:{}", abs_path.display());
    // Escape single quotes in predicate if necessary (simple version for now)
    // LanceDB SQL predicate syntax: "tag = 'value'"
    let predicate = format!("tag = '{}'", tag.replace('\'', "''"));
    // Note: delete might fail if no rows exist, which is fine, but we should handle errors.
    // However, LanceDB delete usually just returns rows deleted or OK.
    let _ = table.delete(&predicate).await;

    // Index new content
    let rows = digest_file(embedder, file_path)?;
    if !rows.is_empty() {
        add_rows(table, rows).await?;
    }

    // Update state
    state.files.insert(
        abs_path,
        FileMetadata {
            mtime_secs: mtime,
            size_bytes: size,
            indexed_at: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)?
                .as_secs(),
        },
    );

    Ok(true)
}

fn digest_file(embedder: &mut TextEmbedding, file_path: &Path) -> Result<Vec<MemoryRow>> {
    let abs_path = fs::canonicalize(file_path).unwrap_or(file_path.to_path_buf());
    let content = fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read {}", file_path.display()))?;

    let chunks = chunk_text(&content, 500, 50);
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let docs: Vec<String> = chunks
        .iter()
        .map(|chunk| format!("passage: {chunk}"))
        .collect();
    let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
    let embeddings = embedder.embed(refs, None)?;

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();
    let mut rows = Vec::with_capacity(chunks.len());
    for (chunk, vector) in chunks.into_iter().zip(embeddings.into_iter()) {
        rows.push(MemoryRow {
            vector,
            text: chunk,
            tag: format!("code:{}", abs_path.display()),
            memory_type: "code".to_string(),
            created_at: now,
        });
    }

    Ok(rows)
}

fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if chunk_size == 0 {
        return Vec::new();
    }

    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;

    let step = chunk_size.saturating_sub(overlap).max(1);
    while start < bytes.len() {
        let end = (start + chunk_size).min(bytes.len());
        chunks.push(String::from_utf8_lossy(&bytes[start..end]).to_string());
        if end == bytes.len() {
            break;
        }
        start += step;
    }

    chunks
}

fn walk_repo_files(root_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    let extensions: HashSet<&'static str> = [
        "py", "rs", "js", "ts", "md", "txt", "nix", "go", "c", "cpp",
    ]
    .into_iter()
    .collect();

    for entry in walkdir::WalkDir::new(root_dir).into_iter().filter_map(Result::ok) {
        let path = entry.path();

        if should_ignore(path) {
            continue;
        }

        if entry.file_type().is_file() {
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            if extensions.contains(ext) {
                paths.push(path.to_path_buf());
            }
        }
    }

    Ok(paths)
}

fn should_ignore(path: &Path) -> bool {
    let ignores: HashSet<&'static str> = [
        ".git",
        "__pycache__",
        "venv",
        "node_modules",
        ".mem_db",
        ".DS_Store",
        "target",
        "dist",
        "result",
    ]
    .into_iter()
    .collect();

    path.components().any(|c| {
        let part = c.as_os_str().to_string_lossy();
        ignores.contains(part.as_ref())
    })
}

fn contains_brackets(text: &str) -> bool {
    text.contains('[') && text.contains(']')
}

async fn ask_table(table: &lancedb::Table, query: &[f32], limit: usize) -> Result<Vec<AskJsonResult>> {
    let stream = table
        .query()
        .nearest_to(query)
        .context("Failed to build vector query")?
        .limit(limit)
        .execute()
        .await?;

    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    let rows = decode_rows(&batches, Some(limit))?;

    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(idx, (tag, text, memory_type, created_at))| AskJsonResult {
            rank: idx + 1,
            tag,
            text,
            memory_type,
            created_at,
        })
        .collect())
}

fn decode_rows(
    batches: &[RecordBatch],
    limit: Option<usize>,
) -> Result<Vec<(Option<String>, String, String, u64)>> {
    let mut out = Vec::new();

    for batch in batches {
        let tag_col = batch
            .column_by_name("tag")
            .context("Missing 'tag' column")?;
        let text_col = batch
            .column_by_name("text")
            .context("Missing 'text' column")?;
        let type_col = batch
            .column_by_name("memory_type")
            .context("Missing 'memory_type' column")?;
        let time_col = batch
            .column_by_name("created_at")
            .context("Missing 'created_at' column")?;

        let tags = tag_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Expected 'tag' to be Utf8 StringArray")?;
        let texts = text_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Expected 'text' to be Utf8 StringArray")?;
        let types = type_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Expected 'memory_type' to be Utf8 StringArray")?;
        let times = time_col
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .context("Expected 'created_at' to be UInt64Array")?;

        for row in 0..batch.num_rows() {
            let tag = if tags.is_null(row) {
                None
            } else {
                Some(tags.value(row).to_string())
            };

            let text = if texts.is_null(row) {
                String::new()
            } else {
                texts.value(row).to_string()
            };

            let mem_type = if types.is_null(row) {
                "unknown".to_string()
            } else {
                types.value(row).to_string()
            };

            let created_at = times.value(row);

            out.push((tag, text, mem_type, created_at));
            if let Some(limit) = limit {
                if out.len() >= limit {
                    return Ok(out);
                }
            }
        }
    }

    Ok(out)
}
