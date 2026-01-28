import json
import os
import fnmatch

import typer
import lancedb
from lancedb.pydantic import LanceModel, Vector
from sentence_transformers import SentenceTransformer
from rich.console import Console

# Initialize the rich console for pretty printing
console = Console()
app = typer.Typer()

# --- CONFIG ---
DB_PATH = "./.mem_db"
MODEL_NAME = "all-MiniLM-L6-v2"

# --- SCHEMA ---
class Memory(LanceModel):
    vector: Vector(384)
    text: str
    tag: str

# --- GLOBALS (Lazy Load) ---
_model = None

def get_model(*, quiet: bool = False):
    """Loads the AI model only when needed (saves startup time)."""
    global _model
    if _model is None:
        if not quiet:
            console.print("[dim]⚡ Loading AI Model...[/dim]")
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def get_table(*, quiet: bool = False):
    """Connects to the database. Creates it if it doesn't exist."""
    db = lancedb.connect(DB_PATH)
    try:
        return db.open_table("memories")
    except (FileNotFoundError, ValueError):
        if not quiet:
            console.print("[yellow]⚠️  Database/Table not found. Initializing...[/yellow]")
        get_model(quiet=quiet)
        return db.create_table("memories", schema=Memory)

def chunk_text(text: str, chunk_size: int = 500):
    """
    Splits long files into smaller, overlapping 'memories'.
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Move forward, but overlap by 50 chars context
        start += (chunk_size - 50) 
        
    return chunks

def should_ignore(path: str) -> bool:
    ignores = {".git", "__pycache__", "venv", "node_modules", ".mem_db", ".DS_Store", "target", "dist", "result"}
    for part in path.split(os.sep):
        if part in ignores:
            return True
    return False

# --- COMMANDS ---

@app.command()
def save(text: str, tag: str = "manual"):
    """
    Manually save a thought to memory.
    Example: mem save "The API key is in .env"
    """
    model = get_model()
    table = get_table()
    
    vector = model.encode(text)
    
    table.add([Memory(vector=vector, text=text, tag=tag)])
    console.print(f"[bold green]✅ Saved:[/bold green] {text}")

@app.command()
def digest(file_path: str):
    """
    Ingest a single code file, chop it up, and save it.
    Example: mem digest ./src/main.py
    """
    if not os.path.exists(file_path):
        console.print(f"[red]❌ File not found:[/red] {file_path}")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        console.print(f"[red]⚠️ Failed to read {file_path}: {e}[/red]")
        return

    chunks = chunk_text(content)
    model = get_model()
    table = get_table()
    
    data_to_add = []
    for chunk in chunks:
        vector = model.encode(chunk)
        data_to_add.append(Memory(
            vector=vector, 
            text=chunk, 
            tag=f"code:{file_path}"
        ))
    
    table.add(data_to_add)
    console.print(f"[green]✅ Digested {file_path}[/green] ({len(chunks)} chunks)")

@app.command()
def learn(root_dir: str = "."):
    """
    Recursively scan and memorize all code in a folder.
    Example: mem learn .
    """
    console.print(f"[bold]🚀 Scanning {root_dir}...[/bold]")
    
    extensions = {".py", ".rs", ".js", ".ts", ".md", ".txt", ".nix", ".go", ".c", ".cpp"}
    
    files_processed = 0
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d))]
        
        for file in files:
            if not any(file.endswith(ext) for ext in extensions):
                continue
                
            path = os.path.join(root, file)
            digest(path) 
            files_processed += 1

    console.print(f"\n[bold green]🏁 Finished processing {files_processed} files.[/bold green]")

@app.command()
def ask(
    question: str,
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as machine-readable JSON",
    ),
):
    """Find relevant memories or code chunks."""
    model = get_model(quiet=json_output)
    table = get_table(quiet=json_output)

    query_vector = model.encode(question)
    results = table.search(query_vector).limit(3).to_list()

    if json_output:
        payload = {
            "question": question,
            "results": [
                {
                    "rank": i + 1,
                    "tag": r.get("tag"),
                    "text": (r.get("text") or "").strip(),
                }
                for i, r in enumerate(results)
            ],
        }
        print(json.dumps(payload, ensure_ascii=False))
        return

    console.print(f"\n[bold blue]🔍 Results for:[/bold blue] '{question}'")
    for i, r in enumerate(results):
        console.print(f"\n[dim]{i+1}. Tag: {r['tag']}[/dim]")
        console.print(r["text"].strip(), markup=False)


@app.command()
def debug(limit: int = 3):
    """Inspect the local LanceDB memory store."""
    try:
        db = lancedb.connect(DB_PATH)
        table = db.open_table("memories")
    except Exception as e:
        console.print(f"[bold red]CRITICAL FAIL:[/bold red] Cannot open DB. {e}")
        return

    count = len(table)
    console.print(f"[bold]Total Memories:[/bold] {count}")

    if count == 0:
        console.print("[yellow]Database is empty.[/yellow]")
        return

    rows = table.search().limit(limit).to_list()

    console.print("\n[bold]___ RAW DATA INSPECTION ___[/bold]")

    for index, row in enumerate(rows):
        content = row.get("text") or ""
        tag = row.get("tag")

        if "[" in content and "]" in content:
            status = "[green]Brackets Found[/green]"
        else:
            status = "[red]No Brackets[/red]"

        console.print(f"\n--- Entry {index + 1} ({status}) ---")
        console.print(f"Tag: {tag}")
        console.print("Content:", style="bold underline")
        console.print(content, markup=False)
        console.print("-" * 30)

if __name__ == "__main__":
    app()
