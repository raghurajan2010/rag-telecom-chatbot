"""LCEL RAG chain: merged retrieval -> prompt -> Qwen on Groq -> string stream."""
import json
import os
import time
import uuid
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import LLM_MODEL, LOG_DIR
from retriever import build_retriever, format_context

FALLBACK = ("I'm sorry, I don't have enough information in our knowledge base to answer that. "
            "Please call 611 or use the MyTelecom app for further help.")

SYSTEM_PROMPT = """You are NovaCell's telecom customer care assistant. You help customers with \
connectivity, mobile data, billing, SIM/eSIM, roaming, voice calls, devices and account questions.

Rules:
- Answer ONLY using the information in the CONTEXT below. Never use outside knowledge, and never \
invent prices, fees, policies, timelines or phone numbers that are not in the context.
- If the context does not contain enough information to answer, reply exactly:
  "{fallback}"
- You have no access to the customer's personal account, balance or bill. If asked for personal \
account data, explain that you can't see account details and point them to the MyTelecom app or 611.
- Be concise and friendly. Use numbered steps for troubleshooting instructions.
- Past resolved tickets (TICKETS) describe what fixed similar issues for other customers; present \
them as things that have helped, not as guaranteed outcomes.
- Earlier messages in the conversation are given only so you can understand follow-up questions \
(for example what "that" refers to). Facts must still come from the CONTEXT.

CONTEXT:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), MessagesPlaceholder("history"), ("human", "{question}")]
).partial(fallback=FALLBACK)


def build_llm(streaming: bool = True):
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    from langchain_groq import ChatGroq

    return ChatGroq(model=LLM_MODEL, temperature=0, reasoning_effort="none", streaming=streaming)


def to_messages(history):
    """[{'role','content'}] -> LangChain chat messages."""
    return [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
            for m in history or []]


class TelecomRAG:
    def __init__(self):
        self.retriever = build_retriever()
        self.generate = PROMPT | build_llm() | StrOutputParser()

    def retrieve(self, question: str):
        # The retrieval query is always the raw question; history never changes it (agent PRD FR-49).
        return self.retriever.invoke(question)

    def stream_answer(self, question: str, docs, history=None):
        """Yield answer tokens grounded in the given retrieved docs."""
        yield from self.generate.stream({"question": question, "context": format_context(docs),
                                         "history": to_messages(history)})


def source_list(docs):
    return [
        {"source": d.metadata.get("source"), "ref": d.metadata.get("ref"),
         "title": d.metadata.get("title"), "snippet": d.page_content.strip()}
        for d in docs
    ]


# ---------------- interaction log (FR-05a) ----------------
LOG_FILE = LOG_DIR / "interactions.jsonl"


def log_event(event: dict) -> str:
    LOG_DIR.mkdir(exist_ok=True)
    event.setdefault("id", uuid.uuid4().hex)
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event["id"]


def log_interaction(question, answer, docs, latency_s, first_token_s, channel,
                    intent=None, trace=None):
    return log_event({
        "type": "interaction", "channel": channel, "question": question, "answer": answer,
        "intent": intent, "trace": list(trace or []),
        # True only when the bot couldn't answer from context (PRD deflection metric).
        "escalated": "don't have enough information" in answer,
        "mentions_611": "611" in answer,"sources": [d.metadata.get("ref") for d in docs],
        "latency_s": round(latency_s, 3), "first_token_s": round(first_token_s, 3),
    })


def log_feedback(interaction_id, rating):
    log_event({"type": "feedback", "interaction_id": interaction_id, "rating": rating})


class Timer:
    def __init__(self):
        self.start = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.start
