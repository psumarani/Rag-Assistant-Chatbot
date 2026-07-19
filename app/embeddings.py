"""Embeddings module.

Sole responsibility: provide a single, cached embedding model instance
(sentence-transformers, run locally — no API cost) for use by
`vector_store.py` when building or querying the FAISS index.
"""

from __future__ import annotations

import functools

from langchain_huggingface import HuggingFaceEmbeddings

from app.logger import get_logger
from config.constants import DEFAULT_EMBEDDING_MODEL

logger = get_logger(__name__)


@functools.lru_cache(maxsize=1)
def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Return a cached sentence-transformers embedding model.

    The `lru_cache` ensures the (relatively expensive to load) model
    weights are only initialized once per process, regardless of how
    many times this function is called.

    Args:
        model_name: HuggingFace model identifier to load.

    Returns:
        A configured `HuggingFaceEmbeddings` instance ready for use
        with FAISS.
    """
    logger.info("Loading embedding model: %s", model_name)
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    logger.info("Embedding model '%s' loaded successfully", model_name)
    return model
