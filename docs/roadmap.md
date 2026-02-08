# Mandrid Evolution Plan

Mandrid is a high-performance, local-first AI memory layer designed to outperform ByteRover and Cipher by prioritizing CLI efficiency, structural code understanding, and episodic reasoning traces.

## 🛠 Features Implemented

- [x] **Rust-powered Core:** High performance, low footprint.
- [x] **LanceDB Integration:** Vector database with Full-Text Search (FTS).
- [x] **AST-Aware Chunking (Tree-sitter):** Understands Rust, Python, and JS structure (Functions, Structs, Impls).
- [x] **Structural Context Propagation:** Chunks are prefixed with their parent scope (e.g., `[Context: struct Cli]`).
- [x] **Hybrid Search (Vector + FTS):** Combines "vibes" with deterministic keyword matching.
- [x] **RRF Reranking:** Reciprocal Rank Fusion to balance vector and keyword results.
- [x] **Consolidated State:** Killed `index_state.json`. File metadata now lives in the DB.
- [x] **Session Engine:** Isolated context via `session_id`.
- [x] **Project Briefing:** Integrated into `mem context --human`.
- [x] **Architecture Map (`mem map`):** Provides a high-level symbol map of the codebase.
- [x] **Nix Integration:** Reproducible dev environment.
- [x] **Beads-inspired Task Engine (`mem task`):** Dependency-aware task graph management.
- [x] **Role System:** Configurable agent roles (`programmer` vs `assistant`) with enforced constraints.
- [x] **Task-Aware Context Injection:** `mem context` automatically highlights active goals and dependencies.
- [x] **Task-Aware Retrieval:** `mem ask --task-aware` boosts results based on the current active task description.
- [x] **Cross-Encoder Reranking:** Integrated `BGE-Reranker-Base` via FastEmbed for hyper-precision (`mem ask --rerank`).
- [x] **Filesystem Watcher (`mem watch`):** Automatically re-indexes files on save.
- [x] **LSP Sidecar (`mem lsp`):** A minimal LSP server that receives real-time code changes from your editor (even unsaved buffers).
- [x] **Modular Architecture:** Refactored into `db`, `chunker`, and `task` modules for scalability.
- [x] **Automated Command Tracing (`mem run`):** Executes a command and automatically records stdout/stderr/status to episodic memory.

## 🚀 Features in Progress

- [x] **Implicit Trace-Streaming:** Integrating `mem run` logic into shell hooks for zero-effort capture.
- [x] **Negative Memory:** Remembering "what failed" to avoid repeating mistakes (surfaced in `mem context`).
- [x] **Knowledge Graph Layer:** Basic symbol-reference tracking for blast radius analysis (`mem impact`).
- [x] **Polyglot Expansion:** Structural support for Go, C/C++, Java, C# in addition to Rust/Python/JS/TS.

## 📅 Potential / Future Features

- [ ] **Structured References:** Move from string-based reference matching to structured type-aware tracking.
- [ ] **Team Sync (Optional):** Git-native memory sharing (JSONL traces in `.mandrid/`).
- [ ] **Context Window Optimization:** Better summarization for extremely long reasoning traces.
- [ ] **Dashboard Visualization:** Graph view of the symbol relationships.

## 🖥 Resource Strategy
- **Target:** Systems with Integrated Graphics & High RAM.
- **Model Choice:** Use FastEmbed (ONNX) for CPU-bound embedding and reranking. Avoid LLMs for indexing/retrieval tasks.
- **Efficiency:** Minimize token usage by using high-density Markdown instead of verbose JSON for AI-to-CLI communication.
