use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
            let row = MemoryRow {
                vector: embedding,
                text,
                tag,
            };
            add_rows(&table, vec![row]).await?;
            println!("Saved");
        }
        Command::Digest { file_path } => {
            let mut embedder = init_embedder(cache_dir.clone(), true)?;
            let table = open_or_create_table(&db_path).await?;

            let rows = digest_file(&mut embedder, &file_path).with_context(|| {
                format!("Failed to digest file {}", file_path.display())
            })?;
            add_rows(&table, rows).await?;
            println!("Digested {}", file_path.display());
        }
        Command::Learn { root_dir } => {
            let mut embedder = init_embedder(cache_dir.clone(), true)?;
            let table = open_or_create_table(&db_path).await?;

            let mut processed = 0usize;
            for file in walk_repo_files(&root_dir)? {
                let rows = match digest_file(&mut embedder, &file) {
                    Ok(rows) => rows,
                    Err(err) => {
                        eprintln!("Skipping {}: {err}", file.display());
                        continue;
                    }
                };
                if !rows.is_empty() {
                    add_rows(&table, rows).await?;
                    processed += 1;
                }
            }

            println!("Finished processing {processed} files");
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
            for (idx, (tag, text)) in rows.into_iter().enumerate() {
                let status = if contains_brackets(&text) {
                    "Brackets Found"
                } else {
                    "No Brackets"
                };

                println!("\n--- Entry {} ({}) ---", idx + 1, status);
                println!("Tag: {}", tag.unwrap_or_default());
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

    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(vectors), Arc::new(texts), Arc::new(tags)],
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

    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(vectors), Arc::new(texts), Arc::new(tags)],
    )?)
}

fn digest_file(embedder: &mut TextEmbedding, file_path: &Path) -> Result<Vec<MemoryRow>> {
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

    let mut rows = Vec::with_capacity(chunks.len());
    for (chunk, vector) in chunks.into_iter().zip(embeddings.into_iter()) {
        rows.push(MemoryRow {
            vector,
            text: chunk,
            tag: format!("code:{}", file_path.display()),
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
        .map(|(idx, (tag, text))| AskJsonResult {
            rank: idx + 1,
            tag,
            text,
        })
        .collect())
}

fn decode_rows(
    batches: &[RecordBatch],
    limit: Option<usize>,
) -> Result<Vec<(Option<String>, String)>> {
    let mut out = Vec::new();

    for batch in batches {
        let tag_col = batch
            .column_by_name("tag")
            .context("Missing 'tag' column")?;
        let text_col = batch
            .column_by_name("text")
            .context("Missing 'text' column")?;

        let tags = tag_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Expected 'tag' to be Utf8 StringArray")?;
        let texts = text_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Expected 'text' to be Utf8 StringArray")?;

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

            out.push((tag, text));
            if let Some(limit) = limit {
                if out.len() >= limit {
                    return Ok(out);
                }
            }
        }
    }

    Ok(out)
}
