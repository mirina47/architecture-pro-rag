import json
from datetime import datetime
from pathlib import Path
from rag import rag_pipeline, search

BASE_DIR = Path(__file__).parent.resolve()

GOLDEN_FILE = BASE_DIR / "golden_questions.txt"
LOG_FILE = BASE_DIR / "logs.jsonl"


def load_golden_questions():
    questions = []

    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # split by last question mark
            q_mark = line.rfind("?")
            query = line[:q_mark + 1].strip()
            expected = line[q_mark + 1:].strip()

            questions.append({
                "query": query,
                "expected": expected
            })

    return questions


def is_correct_answer(expected: str, actual: str):
    expected = expected.lower()
    actual = actual.lower()

    if expected.startswith("i don't know"):
        return "i don't know" in actual

    return expected in actual


def evaluate_one(query: str, expected: str):
    timestamp = datetime.utcnow().isoformat()

    # --- retrieval ---
    chunks = search(query)
    chunks_found = len(chunks)
    sources = [c["source"] for c in chunks]

    # --- generation ---
    answer = rag_pipeline(query)
    answer_length = len(answer)

    # --- correctness ---
    correct = is_correct_answer(expected, answer)

    return {
        "timestamp": timestamp,
        "query": query,
        "expected": expected,
        "answer": answer,
        "answer_length": answer_length,
        "chunks_found": chunks_found,
        "sources": sources,
        "correct": correct
    }


def log_result(entry):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    questions = load_golden_questions()

    total = 0
    correct_count = 0

    for item in questions:
        result = evaluate_one(item["query"], item["expected"])
        log_result(result)

        total += 1
        if result["correct"]:
            correct_count += 1

        print(
            f"[{total}] {item['query']}\n"
            f"Expected: {item['expected']}\n"
            f"Chunks: {result['chunks_found']} | "
            f"Correct: {result['correct']}\n"
        )

    accuracy = correct_count / total if total else 0

    print("=" * 40)
    print(f"Total: {total}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
