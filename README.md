# RAG Assistant Chatbot 

A production-ready **Retrieval-Augmented Generation (RAG)** assistant that answers questions strictly from your own uploaded documents — powered by **Google Gemini** and **FAISS**.

> Upload PDFs, DOCX, TXT, or Markdown files → the assistant retrieves relevant passages via semantic search → Gemini generates a grounded answer with cited sources. If the answer isn't in your documents, it says so instead of guessing.

---

## Features

- 📄 Multi-format document support: PDF, DOCX, TXT, Markdown
- 🔍 Semantic search via local `sentence-transformers` embeddings + FAISS
- 🤖 Answer generation via Google Gemini (`gemini-2.5-flash`, free tier)
- 🚫 Strict anti-hallucination prompting — answers only from retrieved context
- 📎 Source citations, similarity scores, and full chunk metadata in the UI
- ➕ Incremental indexing — add documents without rebuilding the whole index
- 💬 Session conversation history with clear/reset controls
- 🛡️ Robust error handling: missing/invalid API keys, corrupted files, network failures
- 🧪 Unit-tested core modules (loader, chunker, retriever, vector store)
- 📝 Centralized logging to `logs/rag_assistant.log`

---

## Architecture

```
Documents (PDF/DOCX/TXT/MD)
        │
        ▼
 document_loader.py  → LangChain Document objects
        │
        ▼
   chunking.py        → RecursiveCharacterTextSplitter, enriched metadata
        │
        ▼
  embeddings.py        → sentence-transformers (all-MiniLM-L6-v2)
        │
        ▼
  vector_store.py       → FAISS index (create / load / save / update / delete)
        │
        ▼
   retriever.py           → similarity search, top-k, threshold, metadata filter
        │
        ▼
   prompts.py               → strict, source-citing, anti-hallucination prompt
        │
        ▼
     llm.py                    → Google Gemini (langchain-google-genai)
        │
        ▼
 rag_pipeline.py                 → orchestrates the full flow
        │
        ▼
ui/streamlit_app.py                → Streamlit UI
```

### Data flow

```
Upload → Load → Split → Embed → Index (FAISS) → [persisted to disk]
                                                        │
User question ──────────► Retrieve top-k chunks ◄──────┘
                                   │
                          Build grounded prompt
                                   │
                         Gemini generates answer
                                   │
                    Answer + Sources + Scores → UI
```

### Sequence diagram (asking a question)

```
User          Streamlit UI       RAGPipeline        FAISS          Gemini
 │  question       │                  │               │               │
 │ ───────────────►│                  │                │               │
 │                 │  answer_question │                │               │
 │                 │ ────────────────►│                │               │
 │                 │                  │ similarity_search              │
 │                 │                  │ ──────────────►│               │
 │                 │                  │ ◄──────────────│ chunks+scores │
 │                 │                  │  build_rag_prompt              │
 │                 │                  │ ──────────────────────────────►│
 │                 │                  │ ◄──────────────────────────────│ answer
 │                 │ ◄────────────────│                │               │
 │ ◄───────────────│ answer + sources │                │               │
```

---

## Folder Structure

```
RAG_Project/
├── app/
│   ├── rag_pipeline.py      # Orchestrates the full RAG flow
│   ├── retriever.py         # Similarity search + structured results
│   ├── llm.py                # Gemini client init + generation
│   ├── prompts.py             # Anti-hallucination prompt templates
│   ├── embeddings.py           # sentence-transformers model loader
│   ├── vector_store.py          # FAISS create/load/save/update/delete
│   ├── chunking.py                # RecursiveCharacterTextSplitter wrapper
│   ├── document_loader.py          # PDF/DOCX/TXT/MD loading
│   ├── memory.py                    # Session conversation history
│   ├── utils.py                      # Shared helpers (validation, timing)
│   └── logger.py                      # Centralized logging setup
├── config/
│   ├── settings.py       # .env-driven configuration (typed, validated)
│   └── constants.py      # Fixed, environment-independent values
├── data/
│   ├── documents/        # Uploaded source files land here
│   └── vector_store/     # Persisted FAISS index
├── logs/                 # Rotating log files
├── tests/                 # pytest unit tests
├── ui/
│   └── streamlit_app.py  # Streamlit front-end
├── requirements.txt
├── .env.example
├── main.py
└── README.md
```

---

## Installation

### 1. Clone and enter the project

```bash
git clone https://github.com/psumarani/Rag-Assistant-Chatbot.git
cd RAG_Project
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

### 1. Get a free Google Gemini API key

Visit [Google AI Studio](https://aistudio.google.com/apikey) and generate a free API key.

### 2. Create your `.env` file

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with your real key:

```env
GOOGLE_API_KEY=your_actual_key_here
GEMINI_MODEL=gemini-2.5-flash
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=4
TEMPERATURE=0.2
MAX_OUTPUT_TOKENS=2048
```

> ⚠️ `.env` is git-ignored 

---

## How to Run

```bash
streamlit run main.py
```

The app will open at `http://localhost:8501`.

---

## Example Usage

1. Open the app and confirm the sidebar shows **"Gemini API key detected"**.
2. Drag and drop one or more PDF/DOCX/TXT/MD files into the uploader.
3. Click **Build Knowledge Base** and wait for the progress indicator.
4. Type a question in the chat box with reference to uploaded source document, e.g. *"What does this document  say about refund policy?"*
5. Review the generated answer along with its **source document**, **similarity score**, and **expandable chunk text**.
6. Use **Clear Chat** to reset conversation history, or **Reset KB** to remove all indexed documents.

---

## Testing

```bash
pytest tests/ -v
```

Covers: document loading, chunking, vector store CRUD, retrieval, and prompt assembly.

---

## Future Improvements

- Hybrid search (BM25 + dense vectors)
- Pluggable vector backends (ChromaDB, Pinecone, Milvus, Redis)
- Reranking models for higher-precision retrieval
- Graph RAG / knowledge graph construction
- Agentic RAG (multi-step retrieval and reasoning)
- OCR support for scanned/image-based documents
- Audio transcript ingestion

---

## Known Limitations

- Embedding model runs on CPU by default (no GPU acceleration configured).
- FAISS similarity scores are approximate (converted from L2 distance) and not a calibrated probability.
- Very large documents may take noticeable time to embed on first indexing.
- Google Gemini free tier has request-rate limits; heavy usage may hit them.

---

## License

MIT License — feel free to use, modify, and build upon this project.

## Contributing

Issues and pull requests are welcome. Please open an issue first to discuss significant changes.
