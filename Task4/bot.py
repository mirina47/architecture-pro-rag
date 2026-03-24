
import time
import requests
import os
from rag import rag_pipeline 
from dotenv import load_dotenv

load_dotenv()

# ================= TELEGRAM =================

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# ================= PROXY =================

PROXY_HOST = os.environ.get("PROXY_HOST")
PROXY_PORT = os.environ.get("PROXY_PORT")

proxy_url = f"socks5h://{PROXY_HOST}:{PROXY_PORT}"

proxies = {
    "http": proxy_url,
    "https": proxy_url,
}

# ================= TELEGRAM API =================

def get_updates(offset=None):
    params = {"timeout": 30}
    if offset:
        params["offset"] = offset

    response = requests.get(
        f"{BASE_URL}/getUpdates",
        params=params,
        proxies=proxies,
        timeout=60,
    )

    return response.json()


def send_message(chat_id, text):
    data = {
        "chat_id": chat_id,
        "text": text[:4000], 
    }

    requests.post(
        f"{BASE_URL}/sendMessage",
        json=data,
        proxies=proxies,
        timeout=60,
    )

# ================= BOT LOOP =================

def main():
    print("Telegram RAG бот запущен...")
    offset = None

    while True:
        try:
            updates = get_updates(offset)

            for update in updates.get("result", []):
                offset = update["update_id"] + 1

                if "message" not in update:
                    continue

                message = update["message"]
                chat_id = message["chat"]["id"]
                text = message.get("text")

                if not text:
                    continue

                print(f"\nUser: {text}")

                # ================= RAG =================
                answer = rag_pipeline(text)

                print(f"Bot: {answer}")

                send_message(chat_id, answer)

        except Exception as e:
            print("Ошибка:", e)
            time.sleep(5)


if __name__ == "__main__":
    main()
