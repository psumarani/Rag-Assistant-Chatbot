"""Application entry point for the Professional RAG Assistant.

Run with:
    streamlit run main.py

This file exists (rather than pointing users directly at
`ui/streamlit_app.py`) so the run command matches standard Python
project conventions and so startup logging happens in one obvious place.
"""

from __future__ import annotations

from app.logger import get_logger
from ui.streamlit_app import main

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Starting Professional RAG Assistant")
    main()
