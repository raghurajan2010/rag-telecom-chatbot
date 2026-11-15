"""Merged retriever: queries every registered collection in parallel.

To add a knowledge source (NFR-06): write an ingest_<name>.py that fills a new
Chroma collection, then add an entry to COLLECTIONS below.
"""
from langchain_core.runnables import RunnableLambda, RunnableParallel

from config import TOP_K, get_vectorstore

# collection name -> label used in the prompt context
COLLECTIONS = {
    "faq": "FAQ",
    "tickets": "TICKETS",
    "guides": "GUIDES",
}


def build_retriever(k: int = TOP_K):
    """Return a runnable: question -> list[Document] (k per collection, source-labelled)."""
    branches = {
        name: get_vectorstore(name).as_retriever(search_kwargs={"k": k})
        for name in COLLECTIONS
    }

    def merge(results: dict):
        docs = []
        for name, label in COLLECTIONS.items():
            for doc in results[name]:
                doc.metadata["collection_label"] = label
                docs.append(doc)
        return docs

    return RunnableParallel(branches) | RunnableLambda(merge)


def format_context(docs) -> str:
    blocks = []
    for doc in docs:
        label = doc.metadata.get("collection_label", "?")
        ref = doc.metadata.get("ref", "")
        blocks.append(f"[{label} | {ref}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)
