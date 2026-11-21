"""Orchestrator: intent detection, then the existing RAG pipeline or the Refund Agent.

    User message
        │  (a pending refund confirmation is checked first, deterministically)
        ▼
    Intent detection ──requirement_inquiry──▶ TelecomRAG (unchanged retrieval + history)
        │
        └──refund_request──▶ RefundAgent (tools, validation, confirmation, simulated action)
"""
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from account_data import AccountStore
from intent import REFUND, REQUIREMENT, classify_intent
from rag_chain import TelecomRAG, build_llm
from refund_agent import RefundAgent, Trace, parse_confirmation

HISTORY_EXCHANGES = 4  # FR-50: only the last 4 user/assistant exchanges are passed as context

logger = logging.getLogger("novacell.agent")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@dataclass
class Reply:
    route: str                       # "rag" or "refund"
    intent: str | None
    trace: list
    text: str | None = None          # refund replies
    stream: Iterator[str] | None = None  # rag replies (token stream)
    docs: list = field(default_factory=list)
    confirmation: dict | None = None  # refund waiting for "Yes, proceed" / "Cancel"
    note: str | None = None           # reminder that a refund is still pending


class ChatSession:
    """Everything that belongs to one user's conversation (FR-45..48)."""

    def __init__(self, llm, preset: str = "default"):
        self._llm = llm
        self.history: list[dict] = []
        self.reset_account(preset)

    def reset_account(self, preset: str = "default"):
        self.store = AccountStore(preset)
        self.agent = RefundAgent(self._llm, self.store)

    def recent_history(self) -> list[dict]:
        return self.history[-2 * HISTORY_EXCHANGES:]

    def add_exchange(self, user_text: str, assistant_text: str):
        self.history += [{"role": "user", "content": user_text},
                         {"role": "assistant", "content": assistant_text}]

    def clear(self):
        """Clear conversation: forget history and any pending refund (FR-48, UI section 11)."""
        self.history = []
        self.agent.reset()


class SupportAssistant:
    def __init__(self):
        self.rag = TelecomRAG()
        self.llm = build_llm(streaming=False)  # same model/config as RAG (FR-04)

    def new_session(self, preset: str = "default") -> ChatSession:
        return ChatSession(self.llm, preset)

    def handle(self, session: ChatSession, message: str) -> Reply:
        trace = Trace(logger)
        history = session.recent_history()
        agent = session.agent

        # An explicit "Yes, proceed" / "Cancel" answers a pending confirmation directly.
        if agent.state.status == "awaiting_confirmation":
            decision = parse_confirmation(message)
            if decision:
                trace.append(f"[Intent] {REFUND} (confirmation reply)")
                r = agent.handle_confirmation(decision, trace)
                return Reply("refund", REFUND, trace, text=r.text, confirmation=r.confirmation)

        pending = agent.pending_description()
        try:
            intent = classify_intent(self.llm, message, history, pending)
        except Exception as e:  # classifier unavailable: never trigger an action by accident
            intent = REFUND if pending else REQUIREMENT
            trace.append(f"[Intent] classifier failed ({type(e).__name__}); defaulting")
        trace.append(f"[Intent] {intent}")

        if intent == REFUND:
            r = agent.handle_request(message, history, trace)
            return Reply("refund", intent, trace, text=r.text, confirmation=r.confirmation)

        docs = self.rag.retrieve(message)
        trace.append(f"[RAG] retrieved {len(docs)} documents")
        note = None
        if agent.state.pending:
            note = f"Your refund is still waiting for you to {pending}."
        return Reply("rag", intent, trace, stream=self.rag.stream_answer(message, docs, history),
                     docs=docs, confirmation=self._pending_confirmation(agent), note=note)

    @staticmethod
    def _pending_confirmation(agent: RefundAgent) -> dict | None:
        s = agent.state
        if s.status == "awaiting_confirmation":
            return {"amount": s.amount, "action": s.action}
        return None
