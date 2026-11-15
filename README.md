# 📶 NovaCell Telecom Support Chatbot — a RAG project

A customer-support chatbot for a fictional mobile operator, NovaCell. It answers everyday (Tier-1) questions about mobile data, billing, SIM/eSIM, roaming, calls and accounts, **using only the company's own knowledge**: an FAQ, a database of past resolved support tickets, and a PDF user guide.

It uses **Retrieval-Augmented Generation (RAG)**: for each question it first searches the three sources for relevant passages, then asks an LLM to answer *only* from those passages and to show its sources. If the answer isn't in the knowledge base, it says so and points the customer to a human (call 611) instead of making something up.

Built from a product requirements document: [PRD.md](PRD.md) (problem brief: [PROBLEM_STATEMENT.txt](PROBLEM_STATEMENT.txt)).

## Features

- 💬 **Chat web app** (Streamlit) with streamed answers, clickable sample questions and a "Clear conversation" button
- 📚 **Searches three sources in parallel**: FAQ, resolved tickets and PDF guide, taking the top 3 matches from each
- 🔎 **Sources shown with every answer** (e.g. `FAQ-10`, `TK-004`, `telecom_guide.pdf p.5`)
- 🛡️ **Answers only from the knowledge base**, with a fixed "I don't have enough information, please call 611" fallback
- 🔒 **Respects privacy limits**: it has no account access, so it explains that instead of guessing a customer's balance or bill
- 👍👎 **Feedback buttons** plus an interaction log (question, answer, sources, response time, whether it had to escalate)
- ⌨️ **CLI mode** for use in a terminal
- 🔁 **Easy to update knowledge**: support staff edit the CSV, database or PDF and re-run one ingest script

## How it works

```
User question
     │
     ▼
Merged retriever (3 searches in parallel)
  ├── ChromaDB · faq       → top-3 FAQ entries
  ├── ChromaDB · tickets   → top-3 resolved tickets
  └── ChromaDB · guides    → top-3 PDF guide chunks (600 chars, 100 overlap)
     │
     ▼  9 passages, each labelled with its source
Prompt: "answer ONLY from this context, else say you don't know"
     │
     ▼
Qwen 27B on Groq (temperature 0) → answer streamed to the UI, with sources
```

## Example results

Live test run against the real model:

| Question | What the bot did | First token |
|---|---|---|
| "Why is my mobile internet so slow?" | Step-by-step fix taken from the FAQ | 0.74 s |
| "I was charged for roaming after coming back from Spain" | Explained using a similar resolved ticket (TK-004); noted it can't see the customer's bill | 0.26 s |
| "What is my current account balance?" | Said it has no account access; pointed to the app / `*123#` | 0.25 s |
| "Who won the 2022 football World Cup?" | Declined: "I don't have enough information… call 611" | 0.25 s |

## Tech stack

| Part | Tool |
|---|---|
| Orchestration | LangChain (LCEL) |
| Vector database | ChromaDB (saved on disk) |
| Embeddings | `all-MiniLM-L6-v2`, run locally via ONNX Runtime (no embedding API cost) |
| LLM | `qwen/qwen3.8-27b` on Groq |
| UI | Streamlit |
| Data | CSV (FAQ), SQLite (tickets), PDF (guide, parsed with pypdf) |

## Run it yourself

Requires Python 3.11+ and a free [Groq API key](https://console.groq.com).

```powershell
git clone <this repo's URL>
cd <repo folder>
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env          # then paste your GROQ_API_KEY into .env
.venv\Scripts\python.exe ingest_all.py
.venv\Scripts\streamlit.exe run app.py
```

Then open http://localhost:8501. For the terminal version, run `.venv\Scripts\python.exe main.py` (type `quit` to exit).

On macOS/Linux, use `.venv/bin/` instead of `.venv\Scripts\` and `cp` instead of `copy`.

## Updating the knowledge base

| Change | Re-run |
|---|---|
| Edit `data/faq.csv` | `python ingest_faq.py` |
| Add resolved tickets to `data/tickets.db` | `python ingest_tickets.py` |
| Replace `data/telecom_guide.pdf` | `python ingest_guides.py` |

Each script rebuilds only its own collection, so it's safe to re-run. Restart the app afterwards. Only tickets with `status = 'resolved'` are used.

**Adding a new source:** write an `ingest_<name>.py` (using `config.rebuild_collection`) and add the collection to `COLLECTIONS` in `retriever.py`.

## Project structure

| File | Purpose |
|---|---|
| `config.py` | Paths, model names, embedding and ChromaDB helpers |
| `ingest_faq.py`, `ingest_tickets.py`, `ingest_guides.py`, `ingest_all.py` | Load each source into `chroma_store/` |
| `retriever.py` | Parallel search across the three collections, with source labels |
| `rag_chain.py` | Prompt, LLM, streaming, and the interaction/feedback log |
| `app.py` | Streamlit web UI |
| `main.py` | Command-line chat |
| `data/` | Sample knowledge sources (FAQ, tickets, guide) |

## What I learned / design decisions

- **Show sources and refuse when unsure.** The prompt allows only retrieved context and gives a fixed fallback message, so made-up answers are easy to spot and count.
- **Search each source separately.** Taking the top 3 from each source stops the FAQ (short, keyword-heavy entries) from crowding out ticket and guide results.
- **Measure escalation precisely.** An answer counts as escalated only when the bot falls back. Helpful answers that also say "call 611 if it persists" don't count against the deflection metric.
- **Fast startup.** Loading MiniLM through PyTorch took minutes on a laptop with antivirus scanning. Switching to the same model on ONNX Runtime brought startup down to a few seconds.
- **Model availability.** The PRD named `qwen3.6-27b`, which Groq no longer serves, so the model is configurable through `LLM_MODEL` in `.env`.

## Possible next steps

Remember earlier questions in the conversation, hybrid search (BM25 + vectors), re-ranking, an automated evaluation set (e.g. RAGAS), and handing unresolved chats to a human agent.

## Notes

- `truststore` makes Python use the operating system's certificates, which helps behind corporate proxies or antivirus that inspect HTTPS traffic.
- The first run downloads the embedding model (~80 MB) to `~/.cache/chroma`. After that, embeddings run offline.
- All company names and data are fictional sample data for this exercise.
