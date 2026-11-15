"""Ingest resolved tickets from data/tickets.db into the `tickets` collection."""
import sqlite3

from langchain_core.documents import Document

from config import TICKETS_DB, rebuild_collection


def load_ticket_docs():
    docs, ids = [], []
    with sqlite3.connect(TICKETS_DB) as conn:
        rows = conn.execute(
            "SELECT ticket_id, category, issue_type, description, resolution "
            "FROM tickets WHERE lower(status) = 'resolved'"
        ).fetchall()
    for ticket_id, category, issue_type, description, resolution in rows:
        docs.append(
            Document(
                page_content=(f"Issue: {issue_type} (category: {category})\n"
                              f"Description: {description}\nResolution: {resolution}"),
                metadata={"source": "TICKET", "ref": ticket_id,
                          "title": issue_type, "category": category},
            )
        )
        ids.append(f"ticket-{ticket_id}")
    return docs, ids


def main():
    rebuild_collection("tickets", *load_ticket_docs())


if __name__ == "__main__":
    main()
