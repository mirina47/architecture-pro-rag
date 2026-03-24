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

# Yandex Cloud
YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY")   
YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID")   
YANDEX_MODEL_URI = f"gpt://{YANDEX_FOLDER_ID}/yandexgpt"

# ================= INIT =================

print("Загрузка эмбеддинг-модели...")
model = SentenceTransformer(MODEL_NAME)

print("Загрузка индекса FAISS...")
index = faiss.read_index(str(INDEX_FILE))

print("Загрузка метаданных...")
with open(META_FILE, "rb") as f:
    metadata = pickle.load(f)

print("Подключение LLM...")

if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
    raise ValueError("Отсутствуют YANDEX_API_KEY и YANDEX_FOLDER_ID в .env")

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

def build_prompt(question: str, chunks: list):
    context = "\n\n".join([c["text"] for c in chunks])

    prompt = f"""
You are a helpful assistant that answers questions using ONLY the provided context.

If the answer is not in the context, say exactly: I don't know.

You must answer in the following format:

Step 1: ...
Step 2: ...
Step 3: ...

Final answer: ...

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

---

Context:
{context}

---

Question:
{question}

Answer:
"""
    return prompt


# ================= LLM (YandexGPT) =================
def ask_llm(prompt: str) -> str:
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    headers = {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "modelUri": YANDEX_MODEL_URI,
        "completionOptions": {
            "stream": False,
            "temperature": 0.2,
            "maxTokens": 700
        },
        "messages": [
            {
                "role": "system",
                "text": (
                    "You are a reasoning assistant. "
                    "Always explain your reasoning step-by-step "
                    "before giving the final answer."
                )
            },
            {"role": "user", "text": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        return result["result"]["alternatives"][0]["message"]["text"]
    else:
        return f"LLM error: {response.text}"

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
