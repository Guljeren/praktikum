"""
indexer.py — отдельный скрипт индексации (запускается один раз).

Что делает:
  1. читает habr_hub_search.json (все разделы/хабы — ключи верхнего уровня);
  2. берёт ПОЛНЫЙ текст статьи (textHtml), а превью (leadData) — только как запасной вариант;
  3. чистит HTML, режет на чанки: размер 1200, перекрытие 200;
  4. строит векторный индекс FAISS и сохраняет чанки для BM25 (гибридный поиск в app.py).

Запуск:  python indexer.py
"""
import json
import os
import pickle
import sys

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

JSON_PATH = os.environ.get("HABR_JSON", "habr_hub_search.json")
FAISS_DIR = "faiss_index"
CHUNKS_PATH = "chunks.pkl"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "cointegrated/rubert-tiny2")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def html_to_text(html: str) -> str:
    """HTML -> чистый текст (без картинок, скриптов, лишних пробелов)."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "figure", "img", "svg"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def build_documents(data: dict) -> list:
    documents = []
    seen_ids = set()  # одна статья может встретиться в нескольких разделах
    for hub, articles in data.items():
        if not isinstance(articles, list):
            continue
        for art in articles:
            art_id = str(art.get("id", "unknown"))
            if art_id in seen_ids:
                continue
            seen_ids.add(art_id)

            title = html_to_text(art.get("titleHtml", ""))
            body = html_to_text(art.get("textHtml", ""))          # полный текст
            if not body:                                          # запасной вариант — превью
                body = html_to_text((art.get("leadData") or {}).get("textHtml", ""))
            if not body and not title:
                continue

            author_data = art.get("author") or {}
            author = author_data.get("fullname") or author_data.get("alias") or "Неизвестный автор"

            metadata = {
                "id": art_id,
                "title": title,
                "author": author,
                "hub": hub,
                "published": art.get("timePublished", ""),
                "url": f"https://habr.com/ru/articles/{art_id}/",
            }
            documents.append(Document(page_content=f"{title}\n\n{body}", metadata=metadata))
    return documents


def main():
    if not os.path.exists(JSON_PATH):
        print(f"Ошибка: файл {JSON_PATH} не найден. Положите его в папку со скриптом.")
        sys.exit(1)

    print("Загрузка JSON...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    hubs = list(data.keys())
    print(f"Разделов в JSON: {len(hubs)}")
    for h in hubs:
        print(f"  - {h}: {len(data[h]) if isinstance(data[h], list) else '?'} статей")
    if len(hubs) != 7:
        print(f"ВНИМАНИЕ: по заданию ожидается 7 разделов, найдено {len(hubs)}.")

    documents = build_documents(data)
    print(f"Статей после очистки и удаления дублей: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    chunks = splitter.split_documents(documents)
    for i, ch in enumerate(chunks):
        ch.metadata["chunk_id"] = i
    print(f"Всего чанков: {len(chunks)} (размер {CHUNK_SIZE}, перекрытие {CHUNK_OVERLAP})")

    print(f"Векторизация ({EMBEDDING_MODEL}) и создание индекса FAISS — это может занять время...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True}
    )
    FAISS.from_documents(chunks, embeddings).save_local(FAISS_DIR)
    print(f"FAISS сохранён в '{FAISS_DIR}'")

    # BM25 строится в app.py из этих же чанков (быстро), поэтому храним только документы
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Чанки для BM25 сохранены в '{CHUNKS_PATH}'")
    print("Готово! Теперь: streamlit run app.py")


if __name__ == "__main__":
    main()