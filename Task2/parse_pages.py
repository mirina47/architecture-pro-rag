import cloudscraper
from bs4 import BeautifulSoup
from pathlib import Path
import time
import re

BASE_URL = "https://harrypotter.fandom.com/wiki/"

PAGES = [
    "Harry_Potter", "Hermione_Granger", "Ronald_Weasley", "Ginevra_Weasley", "Neville_Longbottom", "Luna_Lovegood", "Tom_Riddle", "Draco_Malfoy", "Albus_Dumbledore", "Severus_Snape",
    "Expecto_Patronum", "Accio", "Wingardium_Leviosa", "Expelliarmus", "Lumos", "Alohomora", "Avada_Kedavra", "Sectumsempra", "Obliviate", "Riddikulus",
    "First_World_War", "Global_Wizarding_War", "First_Wizarding_War", "Second_Wizarding_War", "Halloween_feast", "Acromantula", "Basilisk", "Centaur", "Dragon", "Pixie",
    "Hogwarts_School_of_Witchcraft_and_Wizardry", "Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin", "British_Ministry_of_Magic", "12_Grimmauld_Place", "Death_Eaters", "Dumbledores_Army", "Order_of_the_Phoenix"
]

scraper = cloudscraper.create_scraper()

BASE_DIR = Path(__file__).parent.resolve()
out_dir = BASE_DIR / "pages"
out_dir.mkdir(exist_ok=True)

for page in PAGES:
    url = BASE_URL + page
    r = scraper.get(url)
    print(page, r.status_code)

    soup = BeautifulSoup(r.text, "html.parser")
    content = soup.select_one(".mw-parser-output")
    if not content:
        print("no content", page)
        continue

    # Удаление лишних блоков и таблиц
    for selector in [".warningbox", ".navbox", ".infobox", ".reflist", ".toc", ".mw-empty-elt", ".categorylink", "table"]:
        for el in content.select(selector):
            el.decompose()

    # Удаление сносок
    for sup in content.find_all("sup"):
        sup.decompose()

    paragraphs = []

    # Обработка текста из <p>, <h2>, <h3>
    for tag in content.find_all(["p", "h2", "h3"]):
        text = tag.get_text(" ", strip=True)
        if not text:
            continue
        # Удаление лишних данных
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 50:
            continue
        paragraphs.append(text)

    clean_text = "\n\n".join(paragraphs)

    (out_dir / f"{page}.txt").write_text(clean_text, encoding="utf-8")
    time.sleep(1)

print("Цикл завершен")
