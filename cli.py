import typer
import lancedb
from lancedb.pydantic import LanceModel, Vector
from sentence_transformers import SentenceTransformer
import os
import fnmatch
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

def get_model():
    """
    Loads the AI model only when needed (saves startup time).
    """
    global _model
    if _model is None:
        console.print("[dim]⚡ Loading AI Model...[/dim]")
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def get_table():
    """
    Connects to the database. Creates it if it doesn't exist.
    """
    db = lancedb.connect(DB_PATH)
    try:
        return db.open_table("memories")
    except (FileNotFoundError, ValueError):
        console.print("[yellow]⚠️  Database/Table not found. Initializing...[/yellow]")
        get_model() # Load model to ensure schema is valid
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
def ask(question: str):
    """
    Find relevant memories or code chunks.
    Example: mem ask "How does the chunker work?"
    """
    model = get_model()
    table = get_table()
    
    # Embed the user's question
    query_vector = model.encode(question)
    
    # Search the vector DB
    results = table.search(query_vector).limit(3).to_list()
    
    console.print(f"\n[bold blue]🔍 Results for:[/bold blue] '{question}'")
    for i, r in enumerate(results):
        console.print(f"\n[dim]{i+1}. Tag: {r['tag']}[/dim]")
        
        # This prevents Rich from hiding code like [start:end]
        console.print(r['text'].strip(), markup=False)

if __name__ == "__main__":
    app()
