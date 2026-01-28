import lancedb
from rich.console import Console

console = Console()
DB_PATH = "./.mem_db"

def inspect_brain():
    # 1. Check if DB exists
    try:
        db = lancedb.connect(DB_PATH)
        table = db.open_table("memories")
    except Exception as e:
        console.print(f"[bold red]CRITICAL FAIL:[/bold red] Cannot open DB. {e}")
        return

    # 2. Get Raw Stats
    count = len(table)
    console.print(f"[bold]Total Memories:[/bold] {count}")

    if count == 0:
        console.print("[yellow]Database is empty.[/yellow]")
        return

    # 3. Dump the last 3 entries RAW
    # .to_list() returns standard Python dictionaries - no pandas required
    rows = table.search().limit(3).to_list()

    console.print("\n[bold]___ RAW DATA INSPECTION ___[/bold]")

    for index, row in enumerate(rows):
        content = row['text']
        tag = row['tag']
        
        # logic_check: Does the text actually contain brackets?
        if "[" in content and "]" in content:
            status = "[green]Brackets Found[/green]"
        else:
            status = "[red]No Brackets[/red]"
            
        console.print(f"\n--- Entry {index + 1} ({status}) ---")
        console.print(f"Tag: {tag}")
        
        # CRITICAL: We print with markup=False to prove what is actually there
        console.print("Content:", style="bold underline")
        console.print(content, markup=False)
        console.print("-" * 30)

if __name__ == "__main__":
    inspect_brain()
