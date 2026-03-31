from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import time

BASE_DIR = Path(__file__).parent.resolve()

KB_DIR = BASE_DIR.parent / "Task2" / "knowledge_base"
MALICIOUS_FILE = BASE_DIR.parent / "Task5" / "malicious.txt"
INDEX_FILE = BASE_DIR / "faiss.index"
META_FILE = BASE_DIR / "metadata.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Загрузка эмбеддинг-модели...")
model = SentenceTransformer(MODEL_NAME)

print("Загрузка документов...")

documents = []

# ===== Обычные документы =====
for file in KB_DIR.glob("*.txt"):
    text = file.read_text(encoding="utf-8")

    documents.append({
        "text": text,
        "source": file.name
    })

# ===== Вредоносный документ =====
if MALICIOUS_FILE.exists():
    text = MALICIOUS_FILE.read_text(encoding="utf-8")

    documents.append({
        "text": text,
        "source": MALICIOUS_FILE.name
    })

    print("⚠ Добавлен вредоносный файл:", MALICIOUS_FILE.name)

print("Документы загружены:", len(documents))

print("Разбиение документов на чанки...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80
)

chunks = []

for doc in documents:
    parts = splitter.split_text(doc["text"])

    for i, chunk in enumerate(parts):
        chunks.append({
            "text": chunk,
            "source": doc["source"],
            "chunk_id": i
        })

print("Чанки созданы:", len(chunks))

texts = [c["text"] for c in chunks]

print("Генерация эмбеддингов...")

start_time = time.time()

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)

generation_time = time.time() - start_time

dimension = embeddings.shape[1]

print("Эмбеддинги сгенерированы:", dimension)

print("Создание индекса FAISS...")

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

faiss.write_index(index, str(INDEX_FILE))

with open(META_FILE, "wb") as f:
    pickle.dump(chunks, f)

print("Индекс создан:", INDEX_FILE)
print("Метаданные созданы:", META_FILE)

print("Время генерации:", round(generation_time, 2), "секунд")
