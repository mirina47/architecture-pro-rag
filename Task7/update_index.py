import json
import time
import hashlib
import logging
from pathlib import Path
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- paths ---
BASE_DIR = Path(__file__).parent.resolve()
KB_DIR = BASE_DIR.parent / "Task7" / "knowledge_base"

INDEX_FILE = BASE_DIR / "faiss.index"
META_FILE = BASE_DIR / "metadata.pkl"
STATE_FILE = BASE_DIR / "state.json"
LOG_FILE = BASE_DIR / "update.log"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- logging ---
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def log(msg):
    print(msg)
    logging.info(msg)

# --- utils ---
def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_state():
    if not STATE_FILE.exists():
        return {}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

def load_index():
    if INDEX_FILE.exists():
        return faiss.read_index(str(INDEX_FILE))
    return None

def load_metadata():
    if META_FILE.exists():
        with open(META_FILE, "rb") as f:
            return pickle.load(f)
    return []

# --- main ---
def update_index():
    start = time.time()
    log("Index update started")

    model = SentenceTransformer(MODEL_NAME)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )

    state = load_state()

    new_chunks = []
    updated_files = 0

    for file in KB_DIR.glob("*.txt"):
        try:
            current_hash = file_hash(file)
        except Exception as e:
            log(f"Failed to read {file.name}: {e}")
            continue

        if state.get(file.name) == current_hash:
            continue  # unchanged

        text = file.read_text(encoding="utf-8", errors="ignore")

        parts = splitter.split_text(text)

        for i, chunk in enumerate(parts):
            new_chunks.append({
                "text": chunk,
                "source": file.name,
                "chunk_id": i
            })

        state[file.name] = current_hash
        updated_files += 1
        log(f"Processed {file.name}: {len(parts)} chunks")

    if not new_chunks:
        log("No changes detected")
        return

    texts = [c["text"] for c in new_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)

    index = load_index()

    if index is None:
        log("Creating new FAISS index")
        index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))

    metadata = load_metadata()
    metadata.extend(new_chunks)

    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)

    save_state(state)

    elapsed = round(time.time() - start, 2)

    log(
        f"index updated at {time.strftime('%Y-%m-%d')}, "
        f"{updated_files} files added, "
        f"{len(new_chunks)} chunks added, "
        f"index size: {index.ntotal}, "
        f"errors: 0"
    )

if __name__ == "__main__":
    try:
        update_index()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
