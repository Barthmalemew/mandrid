# Mandrid (mem) 🦀

**The high-performance, local-first AI memory layer for terminal agents.**

Mandrid is a persistent memory layer designed to give terminal-based AI agents (and human developers) structural understanding, episodic context, and long-horizon memory. It replaces naive search with AST-aware Knowledge Graphs and Hybrid Retrieval.

## 🚀 Key Features

- **Polyglot Structural Engine:** AST-aware chunking for Rust, Python, JavaScript/TypeScript, Go, C/C++, Java, and C# using Tree-sitter.
- **Hybrid Search + Reranking:** Combines Vector Search (LanceDB) with Full-Text Search (FTS), fused via RRF and polished with a local Cross-Encoder reranker.
- **Knowledge Graph Intelligence:** Tracks symbol relationships (who calls whom?) to calculate the "Blast Radius" of code changes.
- **Episodic Capture:** Automatically streams terminal commands, exit codes, and git history into searchable "experience."
- **Negative Memory:** Proactively warns about previous command failures and regression risks.
- **Unix-Brain Philosophy:** CLI-first, pipe-friendly JSON output, and stateless execution.
- **Portable & Local:** Everything runs on your CPU. No cloud dependencies. No mandatory IDE plugins.

## 📦 Installation

### Nix (Recommended)
```bash
nix profile install github:Barthmalemew/mandrid
```

### Prebuilt binaries
Download the appropriate binary from GitHub Releases and put it on your `PATH`:

- Linux: `mandrid-linux-amd64`
- macOS: `mandrid-macos-amd64`
- Windows: `mandrid-windows-amd64.exe`

Releases: https://github.com/Barthmalemew/mandrid/releases

Quick install examples:

```bash
# Linux/macOS (rename to `mem` and put on PATH)
chmod +x ./mandrid-<os>-amd64
sudo mv ./mandrid-<os>-amd64 /usr/local/bin/mem
```

Windows: rename `mandrid-windows-amd64.exe` to `mem.exe` and add its folder to your user/system `PATH`.

### Runtime dependency: ONNX Runtime
Mandrid uses local embedding/reranking via `fastembed` (ONNX Runtime).

- Nix installs and wires this up automatically.
- If you install from source or use a raw release binary outside Nix, you must have the ONNX Runtime shared library available.

Common fixes:

- Linux: install `onnxruntime` (distro package) or place `libonnxruntime.so` on `LD_LIBRARY_PATH`.
- macOS: install `onnxruntime` (Homebrew) or set `ORT_DYLIB_PATH` to the directory containing `libonnxruntime.dylib`.
- Windows: ensure `onnxruntime.dll` is on `PATH`.

### From Source
```bash
cargo install --locked --path .
```

## 🛠 Usage

### Initialize
```bash
mem init --role programmer
```

### Learn your Codebase
```bash
mem learn .
```

### Automated Command Capture (Shell Hook)
Source the hook in your shell config (`.zshrc`, `.bashrc`, or PowerShell `$PROFILE`):
```bash
# Zsh
source <(mem hook zsh)

# PowerShell
Invoke-Expression (& mem hook powershell)
```

### Ask Questions
```bash
mem ask "how does the reranking logic work?" --rerank
```

### Impact Analysis
See what might break if you change a symbol:
```bash
mem impact structural_chunk --depth 3
```

### Memory Portal (Web UI)
Launch the local dashboard to browse memories:
```bash
mem serve --port 3000
```

## 🏗 Architecture

Mandrid is built in **Rust** using:
- **LanceDB:** Serverless vector database for disk-based indexing.
- **Tree-sitter:** High-fidelity polyglot parsing.
- **FastEmbed/ONNX:** Local embedding and reranking models.
- **Axum:** Lightweight dashboard backend.

## 📄 License
MIT / Apache-2.0
