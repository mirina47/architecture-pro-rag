import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path

QUERY = "What school did Zight Qun attend?"

BASE_DIR = Path(__file__).parent.resolve()

INDEX_FILE = BASE_DIR / "faiss.index"
META_FILE = BASE_DIR / "metadata.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Загрузка эмбеддинг-модели...")
model = SentenceTransformer(MODEL_NAME)

print("Загрузка FAISS индекса...")
index = faiss.read_index(str(INDEX_FILE))

print("Загрузка метаданных...")
with open(META_FILE, "rb") as f:
    metadata = pickle.load(f)


print("Запрос:", QUERY)

query_vector = model.encode(
    [QUERY],
    normalize_embeddings=True
)

print("Поиск...")

distances, indices = index.search(query_vector, 5)

print("\nРезультаты:\n")

for i, idx in enumerate(indices[0]):

    chunk = metadata[idx]

    print("Результат", i + 1)
    print("Источник:", chunk["source"])
    print("ID чанка:", chunk["chunk_id"])
    print("Ответ:", chunk["text"][:300])
    print()
