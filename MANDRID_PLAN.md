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
- [x] **Project Briefing (`mem brief`):** Summarizes recent decisions and key patterns.
- [x] **Architecture Map (`mem map`):** Provides a high-level symbol map of the codebase.
- [x] **Nix Integration:** Reproducible dev environment.
- [x] **Beads-inspired Task Engine (`mem task`):** Dependency-aware task graph management.
- [x] **Role System:** Configurable agent roles (`programmer` vs `assistant`) with enforced constraints.
- [x] **Task-Aware Context Injection:** `mem context` automatically highlights active goals and dependencies.
- [x] **Task-Aware Retrieval:** `mem ask --task-aware` boosts results based on the current active task description.

## 🚀 Features in Progress

- [ ] **Cross-Encoder Reranking:** A local, CPU-friendly model (e.g., BGE-Reranker) for hyper-precision.


## 📅 Potential / Future Features

- [ ] **Cross-Encoder Reranking:** A local, CPU-friendly model (e.g., BGE-Reranker) for hyper-precision.
- [ ] **LSP Hooking:** Integrating with Language Server Protocol to see code changes in real-time.
- [ ] **Implicit Trace-Streaming:** Automatically capturing reasoning from terminal errors and edits.
- [ ] **Negative Memory:** Remembering "what failed" to avoid repeating mistakes.
- [ ] **Knowledge Graph Layer:** Storing relationships between functions/modules beyond simple chunks.
- [ ] **Team Sync (Optional):** Git-native memory sharing (JSONL traces in `.mandrid/`).

## 🖥 Resource Strategy
- **Target:** Systems with Integrated Graphics & High RAM.
- **Model Choice:** Use FastEmbed (ONNX) for CPU-bound embedding and reranking. Avoid LLMs for indexing/retrieval tasks.
- **Efficiency:** Minimize token usage by using high-density Markdown instead of verbose JSON for AI-to-CLI communication.
