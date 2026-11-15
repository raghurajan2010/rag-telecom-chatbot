"""Shared configuration for the telecom RAG chatbot."""
import os
from pathlib import Path

from dotenv import load_dotenv

try:  # Use the OS certificate store (needed behind TLS-inspecting proxies/antivirus).
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_store"
LOG_DIR = ROOT / "logs"

FAQ_CSV = DATA_DIR / "faq.csv"
TICKETS_DB = DATA_DIR / "tickets.db"
GUIDE_PDF = DATA_DIR / "telecom_guide.pdf"

# PRD specifies qwen/qwen3.6-27b, which Groq no longer serves; qwen3.8-27b is its successor.
# Override with LLM_MODEL in .env.
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
TOP_K = 3
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

_embeddings = None


class MiniLMEmbeddings:
    """all-MiniLM-L6-v2 run locally via ONNX Runtime (Chroma's packaged export).

    Same model as sentence-transformers/all-MiniLM-L6-v2, but avoids importing torch and
    transformers, which can take minutes to load on machines with on-access antivirus scanning.
    The ~80 MB model is downloaded once to ~/.cache/chroma and then runs fully offline.
    """

    def __init__(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._ef = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts):
        return [[float(x) for x in v] for v in self._ef(list(texts))]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def get_embeddings():
    """Local embedding model (cached singleton)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = MiniLMEmbeddings()
    return _embeddings


def get_vectorstore(collection: str):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def rebuild_collection(collection: str, docs, ids):
    """Idempotent ingest: drop the collection and re-add all docs."""
    store = get_vectorstore(collection)
    store.delete_collection()
    store = get_vectorstore(collection)
    store.add_documents(docs, ids=ids)
    print(f"[{collection}] ingested {len(docs)} documents into {CHROMA_DIR}")
    return store
