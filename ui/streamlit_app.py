"""Streamlit front-end for the Professional RAG Assistant.

Responsible only for presentation and user interaction — all business
logic lives in `app.rag_pipeline.RAGPipeline`. Run with:

    streamlit run main.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Ensure the project root is importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.document_loader import EmptyDocumentError
from app.llm import GeminiGenerationError
from app.logger import get_logger
from app.memory import ConversationMemory
from app.rag_pipeline import RAGPipeline
from app.retriever import RetrievedChunk
from app.utils import FileTooLargeError, UnsupportedFileTypeError
from app.vector_store import EmptyVectorStoreError, VectorStoreNotFoundError
from config.constants import SUPPORTED_EXTENSIONS
from config.settings import MissingAPIKeyError, settings

logger = get_logger(__name__)

st.set_page_config(
    page_title="Professional RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_session_state() -> None:
    """Initialize Streamlit session state on first run."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
    if "kb_stats" not in st.session_state:
        st.session_state.kb_stats = None


def _render_sidebar() -> list:
    """Render the sidebar (upload, build KB, reset) and return uploaded files."""
    pipeline: RAGPipeline = st.session_state.pipeline

    with st.sidebar:
        st.title("📚 RAG Assistant")
        st.caption("Powered by Google Gemini + FAISS")

        try:
            settings.require_api_key()
            st.success("Gemini API key detected", icon="✅")
        except MissingAPIKeyError:
            st.error(
                "No Gemini API key found. Copy '.env.example' to '.env' "
                "and add your key from Google AI Studio.",
                icon="⚠️",
            )

        st.divider()
        st.subheader("1. Upload Documents")
        supported = ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS))
        st.caption(f"Supported formats: {supported}")

        uploaded_files = st.file_uploader(
            "Drag and drop files here",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.write(f"**{len(uploaded_files)} file(s) selected:**")
            for uploaded_file in uploaded_files:
                st.text(f"• {uploaded_file.name}")

        st.subheader("2. Build Knowledge Base")
        build_clicked = st.button(
            "🔨 Build Knowledge Base", type="primary", use_container_width=True
        )

        if build_clicked:
            _handle_build_knowledge_base(uploaded_files, pipeline)

        if st.session_state.kb_stats:
            stats = st.session_state.kb_stats
            st.info(
                f"**Files processed:** {stats.files_processed}\n\n"
                f"**Sections loaded:** {stats.document_sections}\n\n"
                f"**Chunks indexed:** {stats.chunks_created}"
            )
        elif pipeline.has_knowledge_base():
            st.info("An existing knowledge base was found and loaded.")

        st.divider()
        st.subheader("3. Session Controls")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.memory.clear()
                st.rerun()
        with col_b:
            if st.button("♻️ Reset KB", use_container_width=True):
                pipeline.reset_knowledge_base()
                st.session_state.kb_stats = None
                st.session_state.memory.clear()
                st.success("Knowledge base reset.")
                st.rerun()

        st.divider()
        with st.expander("⚙️ Retrieval Settings"):
            st.session_state.top_k = st.slider(
                "Top-K chunks", min_value=1, max_value=10, value=settings.top_k
            )
            st.session_state.similarity_threshold = st.slider(
                "Similarity threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05
            )

    return uploaded_files


def _handle_build_knowledge_base(uploaded_files: list, pipeline: RAGPipeline) -> None:
    """Persist uploaded files to disk and build/update the knowledge base."""
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one document first.")
        return

    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    progress_bar = st.sidebar.progress(0, text="Saving uploaded files...")
    for index, uploaded_file in enumerate(uploaded_files):
        destination = settings.documents_dir / uploaded_file.name
        destination.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(destination)
        progress_bar.progress(
            (index + 1) / len(uploaded_files), text=f"Saved {uploaded_file.name}"
        )

    try:
        with st.sidebar:
            with st.spinner("Building knowledge base (loading, chunking, embedding)..."):
                start = time.perf_counter()
                stats = pipeline.build_knowledge_base(saved_paths)
                elapsed = time.perf_counter() - start
        st.session_state.kb_stats = stats
        st.sidebar.success(f"Knowledge base built in {elapsed:.1f}s", icon="✅")
    except (EmptyVectorStoreError, EmptyDocumentError) as exc:
        st.sidebar.error(f"Could not build knowledge base: {exc}")
    except (UnsupportedFileTypeError, FileTooLargeError) as exc:
        st.sidebar.error(str(exc))
    except Exception as exc:  # noqa: BLE001 — surface any unexpected failure safely
        logger.exception("Unexpected error while building knowledge base")
        st.sidebar.error(f"Unexpected error: {exc}")

    progress_bar.empty()


def _render_source(chunk: RetrievedChunk, index: int) -> None:
    """Render a single retrieved chunk inside an expandable panel."""
    with st.expander(
        f"📄 Source {index + 1}: {chunk.source_file}  "
        f"(similarity: {chunk.similarity_score:.2f})"
    ):
        st.markdown(chunk.content)
        st.caption(
            f"Chunk ID: `{chunk.metadata.get('chunk_id', 'n/a')}` | "
            f"Chunk index: {chunk.metadata.get('chunk_index', 'n/a')} | "
            f"Page: {chunk.metadata.get('page', 'n/a')}"
        )


def _render_chat_history() -> None:
    """Render all prior conversation turns."""
    memory: ConversationMemory = st.session_state.memory
    for turn in memory.turns:
        with st.chat_message("user"):
            st.markdown(turn.question)
        with st.chat_message("assistant"):
            st.markdown(turn.answer)
            if turn.sources:
                st.caption(f"Grounded in {len(turn.sources)} source chunk(s)")
                for index, chunk in enumerate(turn.sources):
                    _render_source(chunk, index)


def _handle_question(question: str, pipeline: RAGPipeline) -> None:
    """Run a question through the pipeline and append the result to memory."""
    memory: ConversationMemory = st.session_state.memory

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving context and generating answer..."):
                result = pipeline.answer_question(
                    question=question,
                    top_k=st.session_state.get("top_k"),
                    similarity_threshold=st.session_state.get("similarity_threshold", 0.0),
                )
            st.markdown(result.answer)
            if result.sources:
                st.caption(f"Grounded in {len(result.sources)} source chunk(s)")
                for index, chunk in enumerate(result.sources):
                    _render_source(chunk, index)
            memory.add_turn(question, result.answer, result.sources)
        except VectorStoreNotFoundError as exc:
            st.warning(str(exc))
        except MissingAPIKeyError as exc:
            st.error(str(exc))
        except GeminiGenerationError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 — surface any unexpected failure safely
            logger.exception("Unexpected error while answering question")
            st.error(f"Unexpected error: {exc}")


def main() -> None:
    """Entry point for the Streamlit application."""
    _init_session_state()
    pipeline: RAGPipeline = st.session_state.pipeline

    _render_sidebar()

    st.header("💬 Ask your documents")

    if not pipeline.has_knowledge_base():
        st.info(
            "👋 Upload one or more documents in the sidebar and click "
            "**Build Knowledge Base** to get started."
        )

    _render_chat_history()

    question = st.chat_input("Ask a question about your uploaded documents...")
    if question:
        _handle_question(question, pipeline)


if __name__ == "__main__":
    main()
