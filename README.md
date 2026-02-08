# Mandrid (mem) 🦀

**A high-performance, local-first structural memory layer for terminal-based development.**

Mandrid provides a persistent context layer for developers and autonomous agents. It combines AST-aware code indexing with episodic command tracking to provide a comprehensive project "brain" that remains entirely on-device.

## 🛠 Core Capabilities

- **Structural Indexing:** AST-aware parsing for Rust, Python, Go, C/C++, Java, C#, and JavaScript/TypeScript using Tree-sitter.
- **Hybrid Retrieval:** Combined Vector Search (LanceDB) and Full-Text Search (FTS), optimized with local Cross-Encoder reranking.
- **Blast Radius Analysis:** Deterministic tracking of symbol relationships to map the impact of code changes.
- **Episodic Logging:** Automated capture of terminal sessions, including exit codes and execution telemetry.
- **Proactive Context:** "Negative memory" system that flags previous failures and anti-patterns during active sessions.
- **Local Runtime:** Zero cloud dependencies. All embeddings and reranking processes run on the local CPU.

## 📦 Installation

### Nix (Recommended)
```bash
nix profile install github:Barthmalemew/mandrid
```

### Prebuilt Binaries
Download the latest binary for your architecture from [GitHub Releases](https://github.com/Barthmalemew/mandrid/releases).

**Quick Setup:**
```bash
# Linux/macOS
chmod +x ./mandrid-<os>-amd64
sudo mv ./mandrid-<os>-amd64 /usr/local/bin/mem

# Windows
# Rename to mem.exe and add the directory to your PATH.
```

### Runtime Dependency: ONNX Runtime
Mandrid utilizes `fastembed` for local inference.
- **Nix:** Managed automatically.
- **Manual:** Ensure the ONNX Runtime shared library (`libonnxruntime.so` / `.dylib` / `onnxruntime.dll`) is in your library path.

## 🚀 Getting Started

1. **Initialize a project:**
   ```bash
   mem init --role programmer
   ```

2. **Index the codebase:**
   ```bash
   mem learn .
   ```

3. **Get a compact project briefing (agent-friendly):**
   ```bash
   mem context --human --compact --scope session
   ```

4. **Enable automated capture (Shell Hook):**
   Add the following to your shell configuration (`.zshrc`, `.bashrc`, or PowerShell `$PROFILE`):
   ```bash
   # Zsh/Bash
   source <(mem hook zsh) # or bash
   
   # PowerShell
   Invoke-Expression (& mem hook powershell)
   ```

5. **Query the memory:**
   ```bash
   mem ask "explain the database connection pooling logic" --rerank
   ```

6. **Analyze impact:**
   ```bash
   mem impact handle_request --depth 2
   ```

## 🏗 Technical Stack

Mandrid is implemented in Rust and leverages:
- **LanceDB:** Serverless vector storage.
- **Tree-sitter:** Multi-language AST parsing.
- **ONNX Runtime:** Local model execution for embeddings and reranking.
- **Axum:** Embedded web server for the visualization dashboard.

## 🔧 Upgrades / Rebuild

If you upgrade Mandrid and hit schema/version errors (or indexing/search feels "off"), rebuild the local DB:

```bash
mem rebuild
```

This backs up `.mem_db` and regenerates it.

## 📄 License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
