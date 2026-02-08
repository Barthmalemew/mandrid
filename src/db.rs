use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::fs;

use anyhow::{Context, Result};
use futures::TryStreamExt;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::index::scalar::FtsIndexBuilder;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding, TextRerank, RerankerModel, RerankInitOptions};

use arrow_array::types::Float32Type;
use arrow_array::{Array, FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};

pub const DEFAULT_TABLE_NAME: &str = "memories";
pub const EMBEDDING_DIMS: i32 = 384;

// Bump this when the on-disk DB needs a rebuild.
pub const DB_FORMAT_VERSION: u32 = 1;
const DB_FORMAT_VERSION_FILE: &str = "FORMAT_VERSION";

#[derive(Debug)]
pub struct MemoryRow {
    pub vector: Vec<f32>,
    pub text: String,
    pub tag: String,
    pub memory_type: String,
    pub file_path: String,
    pub line_start: u32,
    pub session_id: String,
    pub name: String,       // Symbol name (e.g. "main")
    pub references: String, // JSON list of calls/imports
    pub depends_on: String, // JSON list
    pub status: String,     // "pending", "active", "completed", "n/a"
    pub mtime_secs: u64,
    pub size_bytes: u64,
    pub created_at: u64,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct DecodedRow {
    pub tag: Option<String>,
    pub text: String,
    pub memory_type: String,
    pub file_path: String,
    pub line_start: u32,
    pub session_id: String,
    pub name: String,
    pub references: String,
    pub depends_on: String,
    pub status: String,
    pub mtime_secs: u64,
    pub size_bytes: u64,
    pub created_at: u64,
}

pub fn init_embedder(cache_dir: Option<PathBuf>, show_progress: bool) -> Result<TextEmbedding> {
    let cache_dir = cache_dir.or_else(|| dirs::cache_dir().map(|p| p.join("mandrid"))).unwrap_or_else(|| PathBuf::from("./.mem_cache"));
    Ok(TextEmbedding::try_new(InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_cache_dir(cache_dir).with_show_download_progress(show_progress))?)
}

pub fn init_reranker(cache_dir: Option<PathBuf>, show_progress: bool) -> Result<TextRerank> {
    let cache_dir = cache_dir.or_else(|| dirs::cache_dir().map(|p| p.join("mandrid"))).unwrap_or_else(|| PathBuf::from("./.mem_cache"));
    Ok(TextRerank::try_new(RerankInitOptions::new(RerankerModel::BGERerankerBase).with_cache_dir(cache_dir).with_show_download_progress(show_progress))?)
}

pub fn embed_prefixed(embedder: &mut TextEmbedding, prefix: &str, text: &str) -> Result<Vec<f32>> {
    let mut embeddings = embedder.embed(vec![format!("{prefix}: {text}")], None)?;
    embeddings.pop().context("No vector")
}

pub fn memory_schema() -> SchemaRef {
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
        Field::new("name", DataType::Utf8, false),
        Field::new("references", DataType::Utf8, false),
        Field::new("depends_on", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("mtime_secs", DataType::UInt64, false),
        Field::new("size_bytes", DataType::UInt64, false),
        Field::new("created_at", DataType::UInt64, false),
    ]))
}

fn ensure_db_format_version_file(db_path: &Path, enforce: bool) -> Result<()> {
    let version_path = db_path.join(DB_FORMAT_VERSION_FILE);

    // If it's a brand-new install or a DB created before we introduced format
    // versioning, we create the marker file and proceed.
    if !version_path.exists() {
        if let Err(e) = fs::write(&version_path, format!("{}\n", DB_FORMAT_VERSION)) {
            // Non-fatal: keep DB usable even if the marker can't be written.
            eprintln!("Warning: failed to write {}: {}", version_path.display(), e);
        }
        return Ok(());
    }

    if !enforce {
        return Ok(());
    }

    let raw = fs::read_to_string(&version_path).unwrap_or_default();
    let found = raw.trim().parse::<u32>().unwrap_or(0);
    if found != DB_FORMAT_VERSION {
        anyhow::bail!(
            "Mandrid database format version mismatch (found {}, expected {}).\n\
             Run `mem rebuild` to regenerate the database (or delete `.mem_db`).",
            found,
            DB_FORMAT_VERSION
        );
    }
    Ok(())
}

async fn open_or_create_table_inner(
    db_path: &Path,
    enforce_format_version: bool,
    validate_schema: bool,
) -> Result<lancedb::Table> {
    // Ensure the folder exists so FORMAT_VERSION can be written.
    let _ = fs::create_dir_all(db_path);

    let db = lancedb::connect(db_path.to_str().context("Invalid db path")?)
        .execute()
        .await?;

    match db.open_table(DEFAULT_TABLE_NAME).execute().await {
        Ok(table) => {
            // Check our DB format version marker.
            // This is separate from the Arrow schema check below.
            ensure_db_format_version_file(db_path, enforce_format_version)?;

            if validate_schema {
                // Validate schema
                let actual_schema = match table.schema().await {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to read schema: {}. Re-initializing might be required.",
                            e
                        );
                        return Ok(table);
                    }
                };
                let expected_schema = memory_schema();

                let actual_fields: std::collections::HashSet<_> = actual_schema
                    .fields()
                    .iter()
                    .map(|f| f.name())
                    .collect();
                let mut missing = Vec::new();
                for field in expected_schema.fields() {
                    if !actual_fields.contains(field.name()) {
                        missing.push(field.name().to_string());
                    }
                }

                if !missing.is_empty() {
                    anyhow::bail!(
                        "Database schema mismatch. Missing fields: {:?}.\n\
                         The database format has changed to support new features (like Blast Radius and Telemetry).\n\
                         Please run `mem rebuild` (or `rm -rf .mem_db` then `mem init`).",
                        missing
                    );
                }
            }
            Ok(table)
        },
        Err(_) => {
            let schema = memory_schema();
            let empty = empty_record_batch(schema.clone())?;
            let iter = vec![empty].into_iter().map(Ok::<_, ArrowError>);
            let batches = RecordBatchIterator::new(iter, schema);

            let table = db
                .create_table(DEFAULT_TABLE_NAME, Box::new(batches))
                .execute()
                .await?;

            ensure_db_format_version_file(db_path, false)?;
            
            table
                .create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
                .execute()
                .await?;

            Ok(table)
        }
    }
}

pub async fn open_or_create_table(db_path: &Path) -> Result<lancedb::Table> {
    open_or_create_table_inner(db_path, true, true).await
}

// Used for recovery / migration paths where we want to read an older DB even if
// the format version marker is mismatched.
// Open an existing DB table without validating schema/version.
// This is intended for best-effort recovery operations (e.g. `mem rebuild`).
pub async fn open_table_existing_unchecked(db_path: &Path) -> Result<lancedb::Table> {
    let db = lancedb::connect(db_path.to_str().context("Invalid db path")?)
        .execute()
        .await?;
    db.open_table(DEFAULT_TABLE_NAME).execute().await.map_err(|e| {
        anyhow::anyhow!("Failed to open existing table at {}: {}", db_path.display(), e)
    })
}

pub fn decode_rows_relaxed(batches: &[RecordBatch], limit: Option<usize>) -> Result<Vec<DecodedRow>> {
    let mut out = Vec::new();

    for batch in batches {
        let n = batch.num_rows();

        let tags = batch
            .column_by_name("tag")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let texts = batch
            .column_by_name("text")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let types = batch
            .column_by_name("memory_type")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let paths = batch
            .column_by_name("file_path")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let lines = batch
            .column_by_name("line_start")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::UInt32Array>());
        let sessions = batch
            .column_by_name("session_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let names = batch
            .column_by_name("name")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let refs = batch
            .column_by_name("references")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let deps = batch
            .column_by_name("depends_on")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let statuses = batch
            .column_by_name("status")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let mtimes = batch
            .column_by_name("mtime_secs")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::UInt64Array>());
        let sizes = batch
            .column_by_name("size_bytes")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::UInt64Array>());
        let times = batch
            .column_by_name("created_at")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::UInt64Array>());

        // Some very old DBs may have used a different column name.
        let created_at_alt = if times.is_none() {
            batch
                .column_by_name("timestamp")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::UInt64Array>())
        } else {
            None
        };

        for row in 0..n {
            let tag = tags
                .and_then(|a| if a.is_null(row) { None } else { Some(a.value(row).to_string()) });
            let text = texts
                .map(|a| if a.is_null(row) { String::new() } else { a.value(row).to_string() })
                .unwrap_or_default();
            let mem_type = types
                .map(|a| if a.is_null(row) { "unknown".to_string() } else { a.value(row).to_string() })
                .unwrap_or_else(|| "unknown".to_string());
            let path = paths
                .map(|a| if a.is_null(row) { "unknown".to_string() } else { a.value(row).to_string() })
                .unwrap_or_else(|| "unknown".to_string());
            let line = lines.map(|a| a.value(row)).unwrap_or(0);
            let session = sessions
                .map(|a| if a.is_null(row) { "default".to_string() } else { a.value(row).to_string() })
                .unwrap_or_else(|| "default".to_string());
            let name = names
                .map(|a| if a.is_null(row) { String::new() } else { a.value(row).to_string() })
                .unwrap_or_default();
            let reference = refs
                .map(|a| if a.is_null(row) { "[]".to_string() } else { a.value(row).to_string() })
                .unwrap_or_else(|| "[]".to_string());
            let dep = deps
                .map(|a| if a.is_null(row) { "[]".to_string() } else { a.value(row).to_string() })
                .unwrap_or_else(|| "[]".to_string());
            let status = statuses
                .map(|a| if a.is_null(row) { "pending".to_string() } else { a.value(row).to_string() })
                .unwrap_or_else(|| "pending".to_string());
            let mtime = mtimes.map(|a| a.value(row)).unwrap_or(0);
            let size = sizes.map(|a| a.value(row)).unwrap_or(0);
            let created_at = times
                .map(|a| a.value(row))
                .or_else(|| created_at_alt.map(|a| a.value(row)))
                .unwrap_or(0);

            out.push(DecodedRow {
                tag,
                text,
                memory_type: mem_type,
                file_path: path,
                line_start: line,
                session_id: session,
                name,
                references: reference,
                depends_on: dep,
                status,
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

pub async fn open_table(db_path: &Path) -> Result<lancedb::Table> {
    let table = open_or_create_table(db_path).await?;
    Ok(table)
}

pub async fn check_risk(
    table: &lancedb::Table,
    query_vector: &[f32],
) -> Result<Option<String>> {
    let stream = table.query()
        .nearest_to(query_vector)?
        .only_if("memory_type = 'trace' AND status = 'failure'")
        .limit(1)
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    let rows = decode_rows(&batches, None)?;

    if let Some(best) = rows.first() {
        let risk_msg = format!(
            "[Mandrid Warning]: Similar command failed recently.\nCommand: {}\nResult: {}",
            best.name,
            best.text.lines().find(|l| l.contains("Exit:") || l.contains("Status:")).unwrap_or("Failed")
        );
        return Ok(Some(risk_msg));
    }
    Ok(None)
}

pub fn empty_record_batch(schema: SchemaRef) -> Result<RecordBatch> {
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
    let names = StringArray::from_iter_values(std::iter::empty::<&str>());
    let refs = StringArray::from_iter_values(std::iter::empty::<&str>());
    let depends_on_vals = StringArray::from_iter_values(std::iter::empty::<&str>());
    let status_vals = StringArray::from_iter_values(std::iter::empty::<&str>());
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
            Arc::new(names),
            Arc::new(refs),
            Arc::new(depends_on_vals),
            Arc::new(status_vals),
            Arc::new(mtimes),
            Arc::new(sizes),
            Arc::new(created_ats),
        ],
    )?)
}

pub async fn add_rows(table: &lancedb::Table, rows: Vec<MemoryRow>) -> Result<()> {
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

pub async fn ask_hybrid(
    table: &lancedb::Table,
    query_vector: &[f32],
    query_text: &str,
    limit: usize,
    vector_only: bool,
    cache_dir: Option<PathBuf>,
    filter: Option<&str>,
) -> Result<Vec<DecodedRow>> {
    let mut reranker = init_reranker(cache_dir, false)?;
    ask_hybrid_with_reranker(table, query_vector, query_text, limit, vector_only, filter, &mut reranker).await
}

pub async fn ask_rrf(
    table: &lancedb::Table,
    query_vector: &[f32],
    query_text: &str,
    limit: usize,
    vector_only: bool,
) -> Result<Vec<DecodedRow>> {
    use lancedb::index::scalar::FullTextSearchQuery;
    use lancedb::query::{ExecutableQuery, QueryBase};
    use lancedb::rerankers::rrf::RRFReranker;

    let query_builder = table.query().nearest_to(query_vector)?;
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
    decode_rows(&batches, Some(limit))
}

pub async fn ask_rrf_filtered(
    table: &lancedb::Table,
    query_vector: &[f32],
    query_text: &str,
    limit: usize,
    vector_only: bool,
    filter: &str,
) -> Result<Vec<DecodedRow>> {
    use lancedb::index::scalar::FullTextSearchQuery;
    use lancedb::query::{ExecutableQuery, QueryBase};
    use lancedb::rerankers::rrf::RRFReranker;

    let query_builder = table.query().nearest_to(query_vector)?.only_if(filter);
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
    decode_rows(&batches, Some(limit))
}

pub async fn ask_hybrid_with_reranker(
    table: &lancedb::Table,
    query_vector: &[f32],
    query_text: &str,
    limit: usize,
    vector_only: bool,
    filter: Option<&str>,
    reranker: &mut TextRerank,
) -> Result<Vec<DecodedRow>> {
    // Phase A: Hybrid retrieval (Vector + FTS fused via RRF)
    let mut results = if let Some(f) = filter {
        ask_rrf_filtered(table, query_vector, query_text, 50, vector_only, f).await?
    } else {
        ask_rrf(table, query_vector, query_text, 50, vector_only).await?
    };
    if results.is_empty() {
        return Ok(results);
    }

    // Phase B: Cross-encoder reranking
    let documents: Vec<&str> = results.iter().map(|r| r.text.as_str()).collect();
    let reranked = reranker.rerank(query_text, documents.as_slice(), false, None)?;

    let mut indexed_results: Vec<_> = results.drain(..).enumerate().collect();
    indexed_results.sort_by(|(idx_a, _), (idx_b, _)| {
        let score_a = reranked.get(*idx_a).map(|r| r.score).unwrap_or(0.0);
        let score_b = reranked.get(*idx_b).map(|r| r.score).unwrap_or(0.0);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(indexed_results
        .into_iter()
        .map(|(_, r)| r)
        .take(limit)
        .collect())
}

pub async fn find_impacted(
    table: &lancedb::Table,
    symbol_name: &str,
) -> Result<Vec<DecodedRow>> {
    // We search for chunks where 'references' JSON array contains the symbol name.
    // Using a simple LIKE filter for now.
    let filter = format!("references LIKE '%\"{}\"%'", symbol_name.replace('\'', "''"));
    let stream = table.query().only_if(filter).execute().await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    decode_rows(&batches, None)
}

pub fn rows_to_batch(schema: SchemaRef, rows: &[MemoryRow]) -> Result<RecordBatch> {
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
    let names = StringArray::from_iter_values(rows.iter().map(|r| r.name.as_str()));
    let refs = StringArray::from_iter_values(rows.iter().map(|r| r.references.as_str()));
    let depends_on_vals = StringArray::from_iter_values(rows.iter().map(|r| r.depends_on.as_str()));
    let status_vals = StringArray::from_iter_values(rows.iter().map(|r| r.status.as_str()));
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
            Arc::new(names),
            Arc::new(refs),
            Arc::new(depends_on_vals),
            Arc::new(status_vals),
            Arc::new(mtimes),
            Arc::new(sizes),
            Arc::new(created_ats),
        ],
    )?)
}

pub fn decode_rows(
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
        let name_col = batch.column_by_name("name").context("Missing 'name' column")?;
        let ref_col = batch.column_by_name("references").context("Missing 'references' column")?;
        let dep_col = batch.column_by_name("depends_on").context("Missing 'depends_on' column")?;
        let status_col = batch.column_by_name("status").context("Missing 'status' column")?;
        let mtime_col = batch.column_by_name("mtime_secs").context("Missing 'mtime_secs' column")?;
        let size_col = batch.column_by_name("size_bytes").context("Missing 'size_bytes' column")?;
        let time_col = batch.column_by_name("created_at").context("Missing 'created_at' column")?;

        let tags = tag_col.as_any().downcast_ref::<StringArray>().context("Expected 'tag' to be Utf8 StringArray")?;
        let texts = text_col.as_any().downcast_ref::<StringArray>().context("Expected 'text' to be Utf8 StringArray")?;
        let types = type_col.as_any().downcast_ref::<StringArray>().context("Expected 'memory_type' to be Utf8 StringArray")?;
        let paths = path_col.as_any().downcast_ref::<StringArray>().context("Expected 'file_path' to be Utf8 StringArray")?;
        let lines = line_col.as_any().downcast_ref::<arrow_array::UInt32Array>().context("Expected 'line_start' to be UInt32Array")?;
        let sessions = session_col.as_any().downcast_ref::<StringArray>().context("Expected 'session_id' to be Utf8 StringArray")?;
        let names = name_col.as_any().downcast_ref::<StringArray>().context("Expected 'name' to be Utf8 StringArray")?;
        let refs = ref_col.as_any().downcast_ref::<StringArray>().context("Expected 'references' to be Utf8 StringArray")?;
        let deps = dep_col.as_any().downcast_ref::<StringArray>().context("Expected 'depends_on' to be Utf8 StringArray")?;
        let statuses = status_col.as_any().downcast_ref::<StringArray>().context("Expected 'status' to be Utf8 StringArray")?;
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
            let name = if names.is_null(row) { String::new() } else { names.value(row).to_string() };
            let reference = if refs.is_null(row) { "[]".to_string() } else { refs.value(row).to_string() };
            let dep = if deps.is_null(row) { "[]".to_string() } else { deps.value(row).to_string() };
            let status = if statuses.is_null(row) { "pending".to_string() } else { statuses.value(row).to_string() };
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
                name,
                references: reference,
                depends_on: dep,
                status,
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
