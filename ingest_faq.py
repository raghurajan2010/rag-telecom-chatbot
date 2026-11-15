"""Ingest data/faq.csv into the `faq` collection (1 row = 1 document)."""
import csv

from langchain_core.documents import Document

from config import FAQ_CSV, rebuild_collection


def load_faq_docs():
    docs, ids = [], []
    with open(FAQ_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            docs.append(
                Document(
                    page_content=f"Q: {row['question']}\nA: {row['answer']}",
                    metadata={"source": "FAQ", "ref": f"FAQ-{row['id']}",
                              "title": row["question"], "category": row.get("category", "")},
                )
            )
            ids.append(f"faq-{row['id']}")
    return docs, ids


def main():
    rebuild_collection("faq", *load_faq_docs())


if __name__ == "__main__":
    main()
