import os
import faiss
import pickle
import requests        
from pathlib import Path
from sentence_transformers import SentenceTransformer          
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG =================

BASE_DIR = Path(__file__).parent.resolve()

INDEX_FILE = BASE_DIR.parent / "Task3" / "faiss.index"
META_FILE = BASE_DIR.parent / "Task3" / "metadata.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# OpenRouter
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# ================= INIT =================

print("Загрузка эмбеддинг-модели...")
model = SentenceTransformer(MODEL_NAME)

print("Загрузка индекса FAISS...")
index = faiss.read_index(str(INDEX_FILE))

print("Загрузка метаданных...")
with open(META_FILE, "rb") as f:
    metadata = pickle.load(f)

print("Подключение LLM...")


# ================= SEARCH =================

def search(query: str):
    query_vector = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_vector, TOP_K)

    results = []
    for i in indices[0]:
        if i >= 0:
            results.append(metadata[i])

    return results

# ================= PROMPT =================

SYSTEM_PROMPT = """
You are a helpful assistant that answers questions using ONLY the provided context.

If the answer is not in the context, say exactly: I don't know.

You must answer in the following format:

Step 1: ...
Step 2: ...
Step 3: ...

Final answer: ...

There must be an empty line before 'Final answer:'.

Few-shot examples:

Q: What school did Zight Qun attend?
A:
Step 1: I search the context for Zight Qun's education.
Step 2: The document states that Zight Qun attended Sheration.
Step 3: Therefore the answer is Sheration.

Final answer: Sheration

Q: Who defeated Lord Lantihoust?
A:
Step 1: I search the context for battles involving Lord Lantihoust.
Step 2: The document says Zight Qun defeated Lord Lantihoust.
Step 3: Therefore the answer is Zight Qun.

Final answer: Zight Qun
"""


def build_prompt(question: str, chunks: list):
    context = "\n\n".join([c["text"] for c in chunks])

    prompt = f"""
Context:
{context}

---

Question:
{question}

Answer:
"""
    return prompt


# ================= LLM =================

def ask_llm(prompt: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "nvidia/nemotron-3-nano-30b-a3b:free",
        "temperature": 0.2,
        "max_tokens": 700,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    try:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception:
        print("LLM raw response:", response.text)
        return "I don't know"


# ================= PIPELINE =================

def rag_pipeline(question: str):
    print("\nПоиск по базе знаний...")

    chunks = search(question)

    if not chunks:
        return "I don't know"

    prompt = build_prompt(question, chunks)
    answer = ask_llm(prompt)

    # safety check
    if "Final answer:" not in answer:
        return "I don't know"

    return answer


# ================= REPL =================

if __name__ == "__main__":
    print("\nRAG бот запущен. Напишите вопрос (exit для выхода).\n")

    while True:
        question = input("Вы: ")

        if question.lower() in ["exit", "quit"]:
            break

        answer = rag_pipeline(question)

        print("\nБот:", answer)
        print()
