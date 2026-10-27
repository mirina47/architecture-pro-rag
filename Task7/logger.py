import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("logs.jsonl")

def log_query(
    query: str,
    chunks_found: int,
    answer: str,
    sources: list
):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "chunks_found": chunks_found,
        "answer_length": len(answer),
        "sources": sources,
        "success": bool(answer.strip()) and chunks_found > 0
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
