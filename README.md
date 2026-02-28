# OkuuLib — RAG-сервис для кыргызской литературы

REST API для индексации `.docx` книг и ответов на вопросы по их содержимому. Использует гибридный поиск (dense + sparse) через Qdrant и генерацию ответов через OpenAI GPT.

---

## Стек

| Компонент | Технология |
|---|---|
| API | FastAPI |
| Векторная БД | Qdrant |
| Dense embeddings | OpenAI `text-embedding-3-small` (1536 dim) |
| Sparse embeddings | BM25 (`Qdrant/bm25` via fastembed) |
| LLM | OpenAI `gpt-4.1-mini` |
| Формат документов | `.docx` |

---

## Быстрый старт

### 1. Переменные окружения

Создай `.env` в корне проекта:

```env
QDRANT_URL=https://your-qdrant-instance
QDRANT_API_KEY=your_qdrant_api_key
OPENAI_API_KEY=your_openai_api_key

# Опционально — значения по умолчанию уже выставлены в config.py
COLLECTION_NAME=okuulib
```

### 2. Запуск через Docker

```bash
docker build -t okuulib .
docker run --env-file .env -p 8000:8000 okuulib
```

### 3. Запуск локально

```bash
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000
```

Swagger UI после запуска: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Структура проекта

```
src/
├── main.py                          # Точка входа FastAPI
├── config.py                        # Все настройки через pydantic-settings
├── health.py                        # GET /health
├── ingestion/
│   ├── api/
│   │   ├── router.py                # POST /ingest/upload, DELETE /ingest/delete
│   │   └── schemas.py               # Pydantic схемы запросов/ответов
│   ├── chunk.py                     # Разбивка текста на чанки
│   ├── clean.py                     # Очистка .docx от мусора (bizdin.kg)
│   ├── cli.py                       # Консольная индексация книг
│   ├── ingest.py                    # Основная логика: загрузка → чанки → embeddings → Qdrant
│   └── load_docx.py                 # Загрузка .docx через Docx2txtLoader
├── retrieval/
│   ├── api/
│   │   ├── router.py                # POST /retrieval/ask
│   │   └── schemas.py
│   ├── prompts/
│   │   ├── system_prompt.txt        # Системный промпт (на кыргызском)
│   │   ├── rag_prompt.txt           # Шаблон запроса с контекстом
│   │   └── prompt_loader.py
│   └── services/
│       ├── llm_service.py           # Генерация ответа через OpenAI
│       └── rag_service.py           # Оркестрация: поиск → генерация
└── shared/
    ├── embedders/
    │   ├── dense_embedder.py        # OpenAI embeddings (батчинг по 100)
    │   └── sparse_embedder.py       # BM25 через fastembed
    └── qdrant/
        ├── vector_search.py         # Гибридный поиск с RRF fusion
        └── vector_store.py          # CRUD коллекции в Qdrant
```

---

## API

### `GET /health`

Проверка работоспособности.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "message": "OkuuLib is running"
}
```

---

### `POST /ingest/upload`

Загружает `.docx` файл, индексирует его в Qdrant и сохраняет в `data/`.

**Запрос:** `multipart/form-data`

| Поле | Тип | Описание |
|---|---|---|
| `file` | File | `.docx` файл книги |

```bash
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@data/manas.docx"
```

```python
import requests

with open("data/manas.docx", "rb") as f:
    r = requests.post(
        "http://localhost:8000/ingest/upload",
        files={"file": ("manas.docx", f)}
    )
print(r.json())
```

**Ответ `200`:**
```json
{
  "book_name": "manas",
  "message": "Book 'manas' successfully ingested."
}
```

**Ошибки:**
| Код | Причина |
|---|---|
| `400` | Загружен не `.docx` файл |
| `500` | Ошибка при индексации (подробности в логах) |

---

### `DELETE /ingest/delete/{book_name}`

Удаляет все чанки книги из Qdrant и файл из `data/`.

**Параметр пути:** `book_name` — имя книги **без расширения** (например `manas`, не `manas.docx`)

```bash
curl -X DELETE http://localhost:8000/ingest/delete/manas
```

**Ответ `200`:**
```json
{
  "book_name": "manas",
  "deleted_chunks": 1842,
  "message": "Deleted 1842 chunks for book 'manas'."
}
```

**Ошибки:**
| Код | Причина |
|---|---|
| `404` | Книга с таким именем не найдена в индексе |

---

### `POST /retrieval/ask`

Задаёт вопрос по книге. Возвращает ответ на кыргызском языке, сгенерированный на основе найденных фрагментов.

**Запрос:** `application/json`

| Поле | Тип | Описание |
|---|---|---|
| `query` | string | Вопрос (желательно на кыргызском) |
| `book_name` | string | Имя книги без расширения |

```bash
curl -X POST http://localhost:8000/retrieval/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Манас кандай баатыр болгон?", "book_name": "manas"}'
```

```python
import requests

r = requests.post(
    "http://localhost:8000/retrieval/ask",
    json={
        "query": "Манас кандай баатыр болгон?",
        "book_name": "manas"
    }
)
print(r.json()["answer"])
```

**Ответ `200`:**
```json
{
  "answer": "Манас — эл аралык даңктуу баатыр, кыргыз элинин коргоочусу..."
}
```

**Ошибки:**
| Код | Причина |
|---|---|
| `400` | Пустой `query` или `book_name` |

---

## Как работает поиск

```
Запрос пользователя
        │
        ├──► Dense embedding (OpenAI)  ──► Qdrant prefetch (семантика)
        │                                                              │
        └──► Sparse embedding (BM25)   ──► Qdrant prefetch (ключевые слова)
                                                                       │
                                               RRF Fusion (объединение)
                                                                       │
                                                    Топ-10 чанков
                                                                       │
                                                  GPT-4.1-mini + промпт
                                                                       │
                                                    Ответ на кыргызском
```

Гибридный поиск объединяет два сигнала: семантическое сходство (dense) и точное совпадение слов (sparse/BM25). Итоговый рейтинг формируется через **RRF (Reciprocal Rank Fusion)** — стандартный алгоритм слияния ранжированных списков.

---


## Настройки

Все параметры в `src/config.py`, переопределяются через `.env`:

| Параметр | По умолчанию | Описание |
|---|---|---|
| `COLLECTION_NAME` | `okuulib` | Название коллекции в Qdrant |
| `CHUNK_SIZE` | `700` | Размер чанка в символах |
| `CHUNK_OVERLAP` | `100` | Перекрытие между чанками |
| `TOP_K` | `10` | Количество чанков, передаваемых в LLM |
| `PREFETCH_MULTIPLIER` | `3` | Множитель предвыборки для RRF (итого 30 кандидатов) |
| `DENSE_EMBEDDING_MODEL` | `text-embedding-3-small` | Модель OpenAI для embeddings |
| `DENSE_EMBEDDING_BATCH_SIZE` | `100` | Размер батча при индексации |
| `SPARSE_EMBEDDING_MODEL` | `Qdrant/bm25` | BM25 модель через fastembed |
| `LLM_MODEL` | `gpt-4.1-mini` | Модель для генерации ответов |
| `QDRANT_BATCH_SIZE` | `16` | Размер батча при записи в Qdrant |
| `QDRANT_TIMEOUT` | `60` | Таймаут подключения к Qdrant (сек) |