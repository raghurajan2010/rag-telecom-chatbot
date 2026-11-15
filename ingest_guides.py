"""Chunk data/telecom_guide.pdf (600 chars, 100 overlap) into the `guides` collection."""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_SIZE, GUIDE_PDF, rebuild_collection


def load_guide_docs():
    reader = PdfReader(str(GUIDE_PDF))
    pages = [
        Document(page_content=page.extract_text() or "",
                 metadata={"page": i + 1})
        for i, page in enumerate(reader.pages)
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]
    docs, ids = [], []
    for n, chunk in enumerate(chunks, start=1):
        ref = f"{GUIDE_PDF.name} p.{chunk.metadata['page']} #{n}"
        chunk.metadata.update({"source": "GUIDE", "ref": ref, "title": GUIDE_PDF.name})
        docs.append(chunk)
        ids.append(f"guide-{n}")
    return docs, ids


def main():
    rebuild_collection("guides", *load_guide_docs())


if __name__ == "__main__":
    main()
