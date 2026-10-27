import json
from pathlib import Path
import re

BASE_DIR = Path(__file__).parent.resolve()

TERMS_MAP_FILE = BASE_DIR / "terms_map.json"
SRC_DIR = BASE_DIR / "pages"
DST_DIR = BASE_DIR / "knowledge_base"
DST_DIR.mkdir(exist_ok=True)

# Загрузка словаря
with open(TERMS_MAP_FILE, "r", encoding="utf-8") as f:
    terms_map = json.load(f)

print(terms_map)

# Регулярки для поиска слов
regex_map = {}
for orig, repl in terms_map.items():
    regex_map[orig] = re.compile(rf"\b{re.escape(orig)}\b", re.IGNORECASE)

for file_path in SRC_DIR.glob("*.txt"):
    text = file_path.read_text(encoding="utf-8")

    # Замена терминов
    for orig, pattern in regex_map.items():
        text = pattern.sub(terms_map[orig], text)

    # Сохранение в файл
    (DST_DIR / file_path.name).write_text(text, encoding="utf-8")

print("Цикл завершен")
