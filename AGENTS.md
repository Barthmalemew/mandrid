# Mandrid Memory Agent Instructions

This project uses **Mandrid** (`mem`) to store context and reasoning.
Before starting tasks, please verify the memory state.

## Quick Start
1. **Check Context:** Run `mem context --human` to see recent decisions and reasoning.
   - For low-noise context: `mem context --human --compact --scope session`
2. **Search:** Run `mem ask --json "query"` to find relevant code or patterns.
3. **Capture:** When completing a significant task, run:
   `mem capture "Reasoning: why I did this change"`

## When Mandrid Updates
- If you hit DB/schema/version errors (or things feel "off" after upgrading), rebuild:
  `mem rebuild`

## Tools
- `mem learn`: Re-indexes the codebase (run this if you suspect the index is stale).
- `mem capture`: Saves your git diff + reasoning to the permanent record.

Do not edit `.mem_db/` manually.
