"""
app.py — веб-интерфейс RAG-системы по статьям Хабра (Streamlit).

Гибридный поиск: FAISS (смысловой) + BM25 (по словам), объединение через RRF.
Генерация: Groq (LangChain, ChatGroq), цепочка собрана на LCEL.
Интерфейс: тёмный градиент, быстрые вопросы, фильтр по разделам, релевантность,
подсветка слов, статистика, история запросов (сохраняется в history.json).

Запуск:  streamlit run app.py
"""
import html
import json
import os
import pickle
import re
import time
from datetime import datetime

import streamlit as st
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "cointegrated/rubert-tiny2")
FAISS_DIR = "faiss_index"
CHUNKS_PATH = "chunks.pkl"
HISTORY_PATH = "history.json"
MAX_HISTORY = 50
RRF_K = 60  # константа Reciprocal Rank Fusion
APP_VERSION = "v6 • автовыбор моделей Groq"  # по этой строке в боковой панели видно, какой файл запущен

# Запасной список, если получить список моделей у Groq не удалось (порядок = предпочтение)
DEFAULT_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
NOT_CHAT_MARKERS = ("whisper", "orpheus", "guard", "tts", "embed", "playai", "distil")

# Быстрые вопросы (подпись на кнопке, текст вопроса). Замените на свои под ваш датасет.
QUICK_QUESTIONS = [
    ("Физики и программисты", "Почему физики и программисты мыслят одинаково?"),
    ("Про искусственный интеллект", "Какие темы про искусственный интеллект обсуждаются?"),
    ("Про Python", "Что пишут про Python?"),
    ("Карьера в IT", "Какие советы по карьере в IT есть в статьях?"),
]

STOPWORDS = {"почему", "какие", "какой", "какая", "что", "как", "это", "для", "или", "про",
             "есть", "при", "над", "под", "если", "чем", "где", "когда", "кто"}


# ======================= Поиск =======================
def tokenize(text: str) -> list:
    """Слова + грубый «стемминг» (первые 6 символов) — помогает BM25 с русскими окончаниями."""
    return [w[:6] for w in re.findall(r"\w+", text.lower())]


def hybrid_retrieve(query, k, faiss_db, bm25, chunks, hubs=None, fetch_k=20):
    """FAISS + BM25 → объединение по RRF. hubs — множество разделов для фильтра (или None).
    Возвращает список Document с метаданными: rrf, rel (0..100), faiss_rank, bm25_rank."""
    hubs = set(hubs) if hubs else None
    scores = {}  # chunk_id -> {"rrf", "doc", "faiss_rank", "bm25_rank"}

    # 1) семантический поиск (фильтр по хабам — на стороне FAISS)
    flt = (lambda m: m.get("hub") in hubs) if hubs else None
    faiss_docs = faiss_db.similarity_search(
        query, k=fetch_k, filter=flt, fetch_k=fetch_k * 20 if hubs else fetch_k)
    for rank, doc in enumerate(faiss_docs, 1):
        e = scores.setdefault(doc.metadata["chunk_id"], {"rrf": 0.0, "doc": doc})
        e["rrf"] += 1.0 / (RRF_K + rank)
        e["faiss_rank"] = rank

    # 2) лексический поиск
    q_tokens = tokenize(query)
    if q_tokens:
        bm = bm25.get_scores(q_tokens)
        cand = [i for i in range(len(bm)) if not hubs or chunks[i].metadata.get("hub") in hubs]
        top = sorted(cand, key=lambda i: bm[i], reverse=True)[:fetch_k]
        rank = 0
        for idx in top:
            if bm[idx] <= 0:
                continue
            rank += 1
            doc = chunks[idx]
            e = scores.setdefault(doc.metadata["chunk_id"], {"rrf": 0.0, "doc": doc})
            e["rrf"] += 1.0 / (RRF_K + rank)
            e["bm25_rank"] = rank

    ranked = sorted(scores.values(), key=lambda e: e["rrf"], reverse=True)[:k]
    best = ranked[0]["rrf"] if ranked else 1.0
    result = []
    for e in ranked:
        meta = {**e["doc"].metadata, "rrf": e["rrf"], "rel": round(100 * e["rrf"] / best),
                "faiss_rank": e.get("faiss_rank"), "bm25_rank": e.get("bm25_rank")}
        result.append(Document(page_content=e["doc"].page_content, metadata=meta))
    return result


# ======================= Цепочка LangChain (LCEL) =======================
PROMPT = ChatPromptTemplate.from_template(
    "Вы — ассистент-исследователь. Ответьте на вопрос пользователя на русском языке, "
    "опираясь исключительно на предоставленный контекст. Если в контексте нет ответа, "
    "прямо скажите, что не можете ответить на основе данных. Не выдумывайте факты.\n\n"
    "Контекст:\n{context}\n\n"
    "Вопрос: {question}\n\n"
    "Ответ:"
)


def format_context(docs) -> str:
    return "\n\n".join(
        f"[Источник {i}: {d.metadata.get('title')}]\n{d.page_content}" for i, d in enumerate(docs, 1)
    )


def pick_chat_models(model_ids) -> list:
    """Из списка моделей Groq оставляет чат-модели (без речевых, TTS и модерации),
    а предпочтительные ставит в начало."""
    ids = sorted({m for m in model_ids if m and not any(t in m.lower() for t in NOT_CHAT_MARKERS)})
    first = [m for m in DEFAULT_MODELS if m in ids]
    return first + [m for m in ids if m not in first]


def is_model_missing(err: Exception) -> bool:
    """Ошибка «модель не существует / нет доступа» — тогда пробуем другую модель."""
    t = str(err).lower()
    return "model_not_found" in t or "does not exist" in t or "decommissioned" in t


def clean_answer(text: str) -> str:
    """Убираем <think>...</think>, которые выдают reasoning-модели (например, Qwen)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_chain(llm):
    # вход: {"docs": [...], "question": "..."}
    return (
        {"context": RunnableLambda(lambda x: format_context(x["docs"])),
         "question": RunnableLambda(lambda x: x["question"])}
        | PROMPT | llm | StrOutputParser() | RunnableLambda(clean_answer)
    )


# ======================= Вспомогательные функции интерфейса =======================
def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def md_to_html(text: str) -> str:
    """Минимальное преобразование ответа LLM в безопасный HTML: **жирный**, переводы строк."""
    t = esc(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return t.replace("\n", "<br>")


def highlight(text: str, query: str) -> str:
    """Экранирует текст и подсвечивает слова запроса (по началу слова, чтобы ловить окончания)."""
    safe = esc(text)
    stems = []
    for w in re.findall(r"\w+", query.lower()):
        if len(w) >= 4 and w not in STOPWORDS:
            stems.append(re.escape(esc(w[:5] if len(w) > 5 else w[:4])))
    if not stems:
        return safe
    pattern = re.compile(r"\b(" + "|".join(set(stems)) + r")\w*", re.IGNORECASE | re.UNICODE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe)


def fmt_time(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
    except Exception:
        return ""
    today = datetime.now().date()
    if dt.date() == today:
        return f"сегодня, {dt:%H:%M}"
    return f"{dt:%d.%m.%Y, %H:%M}"


def load_history() -> list:
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(history: list) -> None:
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history[:MAX_HISTORY], f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # история — удобство, ошибка записи не должна ронять приложение


def unique_sources(docs) -> list:
    seen, out = set(), []
    for d in docs:
        m = d.metadata
        if m["id"] in seen:
            continue
        seen.add(m["id"])
        out.append({"id": m["id"], "title": m.get("title", ""), "url": m.get("url", ""),
                    "author": m.get("author", ""), "hub": m.get("hub", "")})
    return out


def render_answer(res: dict) -> str:
    return f'<div class="card answer"><h2>💡 Ответ</h2><p>{md_to_html(res["answer"])}</p></div>'


def render_sources(sources: list) -> str:
    rows = "".join(
        f'<a class="src" href="{esc(s["url"])}" target="_blank"><b>{esc(s["title"])}</b>'
        f'<span>{esc(s["author"])} • {esc(s["hub"])} • {esc(s["url"])}</span></a>'
        for s in sources)
    return f'<div class="card"><h2>🔗 Ссылки на статьи</h2>{rows}</div>'


def render_chunks(res: dict) -> str:
    parts = []
    for i, d in enumerate(res["docs"], 1):
        m = d["meta"]
        tags = ""
        if m.get("faiss_rank"):
            tags += f'<em class="tag">FAISS #{m["faiss_rank"]}</em>'
        if m.get("bm25_rank"):
            tags += f'<em class="tag t2">BM25 #{m["bm25_rank"]}</em>'
        rel = int(m.get("rel", 0))
        parts.append(
            f'<div class="chunk"><div class="ch"><b>Фрагмент {i}</b>{tags}'
            f'<div class="rel">релевантность<div class="bar"><i style="width:{rel}%"></i></div><b>{rel}%</b></div></div>'
            f'<div class="pre">{highlight(d["text"], res["q"]).replace(chr(10), "<br>")}</div>'
            f'<div class="cmeta">🔗 <a href="{esc(m["url"])}" target="_blank">{esc(m["title"])}</a> • '
            f'ID {esc(m["id"])} • {esc(m["hub"])} • {esc(m["author"])} • '
            f'{esc(str(m.get("published", ""))[:10])} • чанк №{esc(m["chunk_id"])}</div></div>')
    return f'<div class="card"><h2>📌 Использованные чанки</h2>{"".join(parts)}</div>'


def render_history_items(history: list) -> str:
    items = []
    for h in history:
        ans = h.get("answer", "")
        short = ans[:170] + ("…" if len(ans) > 170 else "")
        links = " ".join(f'<a href="{esc(s["url"])}" target="_blank">🔗 {esc(s["title"][:60])}</a>'
                         for s in h.get("sources", [])[:3])
        full = f'<details><summary>Показать полный ответ</summary><div class="full">{md_to_html(ans)}</div></details>' \
            if len(ans) > 170 else ""
        items.append(
            f'<div class="item"><div class="it">{esc(h.get("q", ""))}<span>{esc(fmt_time(h.get("time", "")))}</span></div>'
            f'<div class="ia">{md_to_html(short)}</div>{full}<div class="is">{links}</div></div>')
    return "".join(items)


def load_css() -> str:
    """Стили лежат в отдельном файле style.css рядом с app.py."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f"<style>{f.read()}</style>"
    except OSError:
        return ""  # без стилей приложение всё равно работает, просто выглядит проще


# ============================== UI ==============================


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_groq_models(key: str) -> list:
    """Список доступных чат-моделей прямо у Groq (обновляется раз в 30 минут)."""
    if not key:
        return []
    try:
        from groq import Groq  # ставится вместе с langchain-groq
        return pick_chat_models([m.id for m in Groq(api_key=key).models.list().data])
    except Exception:
        return []


@st.cache_resource(show_spinner="Загрузка индексов...")
def load_indexes():
    if not (os.path.isdir(FAISS_DIR) and os.path.exists(CHUNKS_PATH)):
        return None
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})
    faiss_db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])
    stats = {
        "hubs": sorted({c.metadata.get("hub", "") for c in chunks}),
        "n_articles": len({c.metadata.get("id") for c in chunks}),
        "n_chunks": len(chunks),
    }
    return faiss_db, bm25, chunks, stats


def submit():
    q = st.session_state.get("query_text", "").strip()
    if q:
        st.session_state.pending = q


def pick(q: str):
    st.session_state.query_text = q
    st.session_state.pending = q


def clear_history():
    st.session_state.history = []
    save_history([])


st.set_page_config(page_title="Habr RAG", page_icon="📚", layout="wide")
st.markdown(load_css(), unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = load_history()
st.session_state.setdefault("pending", None)
st.session_state.setdefault("last", None)
st.session_state.setdefault("query_text", "")

indexes = load_indexes()
if indexes is None:
    st.error("Индексы не найдены. Сначала запустите в терминале: `python indexer.py`")
    st.stop()
faiss_db, bm25, chunks, stats = indexes

# ---------- Боковая панель ----------
with st.sidebar:
    st.markdown("### ⚙️ Настройки")
    api_key = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    st.markdown("[Получить ключ](https://console.groq.com/keys)")
    live_models = fetch_groq_models(api_key)
    model_choice = st.selectbox("Модель Groq", (live_models or DEFAULT_MODELS) + ["Другая..."])
    st.caption(f"Моделей Groq доступно: {len(live_models)}" if live_models
               else "Список моделей Groq не получен — показан запасной")
    model_name = st.text_input("Название модели") if model_choice == "Другая..." else model_choice
    hubs_selected = st.multiselect("Разделы (хабы)", stats["hubs"], placeholder="Все разделы")
    k = st.slider("Сколько фрагментов брать (k)", 1, 10, 5)
    temperature = st.slider("Температура", 0.0, 1.0, 0.2, 0.1)
    st.caption(f"Версия приложения: {APP_VERSION}")
    st.markdown(
        f'<div class="stats">'
        f'<div class="st"><b>{stats["n_articles"]:,}</b><span>статей</span></div>'
        f'<div class="st"><b>{stats["n_chunks"]:,}</b><span>чанков</span></div>'
        f'<div class="st"><b>{len(stats["hubs"])}</b><span>разделов</span></div>'
        f'<div class="st"><b>1200/200</b><span>размер/перекрытие</span></div></div>'.replace(",", " "),
        unsafe_allow_html=True)

# ---------- Заголовок и поиск ----------
st.markdown('<h1 class="title">📚 <span>Habr RAG</span></h1>'
            '<div class="subtitle">Поиск по статьям Хабра • гибридный поиск FAISS + BM25 • LangChain • Groq</div>',
            unsafe_allow_html=True)

with st.form("search_form", clear_on_submit=False):
    c_in, c_btn = st.columns([5, 1.7])
    c_in.text_input("Вопрос", key="query_text", label_visibility="collapsed",
                    placeholder="Задайте вопрос по статьям Хабра…")
    c_btn.form_submit_button("Задать вопрос", on_click=submit)

chip_cols = st.columns([1.1] + [2] * len(QUICK_QUESTIONS))
chip_cols[0].markdown('<div style="opacity:.6;padding-top:.4rem;font-size:.85rem">Попробуйте:</div>',
                      unsafe_allow_html=True)
for i, (label, question) in enumerate(QUICK_QUESTIONS):
    chip_cols[i + 1].button(label, key=f"quick_{i}", on_click=pick, args=(question,), use_container_width=True)

# ---------- Обработка запроса ----------
# pending сбрасывается только ПОСЛЕ завершения: если запрос прервали (обновили страницу,
# поменяли настройку во время работы), он автоматически запустится заново.
if st.session_state.pending:
    q = st.session_state.pending
    if not api_key:
        st.warning("Введите Groq API Key в панели слева.")
        st.session_state.pending = None
    elif not model_name:
        st.warning("Укажите название модели.")
        st.session_state.pending = None
    else:
        t0 = time.time()
        with st.status("Обрабатываю вопрос…", expanded=True) as status:
            try:
                status.write("🔎 Ищу релевантные фрагменты (FAISS + BM25)…")
                docs = hybrid_retrieve(q, k, faiss_db, bm25, chunks, hubs=hubs_selected or None)
                status.write(f"✅ Найдено фрагментов: {len(docs)} (поиск {time.time() - t0:.1f} с)")
                if not docs:
                    st.session_state.last = None
                    status.update(label="Ничего не найдено", state="error")
                    st.warning("Релевантная информация в базе не найдена (попробуйте снять фильтр по разделам).")
                else:
                    status.write("🤖 Генерирую ответ (Groq)…")
                    candidates = [model_name] + [m for m in (live_models or DEFAULT_MODELS) if m != model_name]
                    answer, used_model, last_err = None, None, None
                    for m in candidates:
                        try:
                            llm = ChatGroq(groq_api_key=api_key, model_name=m, temperature=temperature)
                            answer = build_chain(llm).invoke({"docs": docs, "question": q})
                            used_model = m
                            break
                        except Exception as e:
                            last_err = e
                            if not is_model_missing(e):
                                raise  # ключ, лимиты и прочее — показываем как есть
                            status.write(f"⚠️ Модель `{m}` недоступна, пробую следующую…")
                    if answer is None:
                        raise last_err
                    if used_model != model_name:
                        status.write(f"ℹ️ Ответ получен моделью `{used_model}`")
                    sources = unique_sources(docs)
                    st.session_state.last = {
                        "q": q, "answer": answer, "sources": sources,
                        "docs": [{"text": d.page_content, "meta": d.metadata} for d in docs],
                    }
                    st.session_state.history.insert(0, {
                        "q": q, "answer": answer, "sources": sources,
                        "time": datetime.now().isoformat(timespec="seconds")})
                    st.session_state.history = st.session_state.history[:MAX_HISTORY]
                    save_history(st.session_state.history)
                    status.update(label=f"Готово за {time.time() - t0:.1f} с • модель {used_model}", state="complete", expanded=False)
            except Exception as e:  # неверный ключ, лимиты, несуществующая модель, сбой поиска
                status.update(label="Ошибка", state="error")
                st.error(f"Ошибка: {e}")
        st.session_state.pending = None

# ---------- Результат ----------
last = st.session_state.last
if last:
    st.markdown(render_answer(last), unsafe_allow_html=True)
    st.markdown(render_sources(last["sources"]), unsafe_allow_html=True)
    st.markdown(render_chunks(last), unsafe_allow_html=True)

# ---------- История ----------
st.markdown("---")
h_title, h_btn = st.columns([5, 1.3])
h_title.markdown("### 🕘 История запросов")
if st.session_state.history:
    h_btn.button("Очистить", key="clear_hist", on_click=clear_history, use_container_width=True)
    st.markdown(f'<div class="card" style="margin-top:0">{render_history_items(st.session_state.history)}</div>',
                unsafe_allow_html=True)
else:
    st.caption("Пока пусто. Задайте первый вопрос.")