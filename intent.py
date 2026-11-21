"""Intent detection (agent PRD 6.1): requirement_inquiry vs refund_request, via the existing LLM."""
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

REQUIREMENT = "requirement_inquiry"
REFUND = "refund_request"


class IntentResult(BaseModel):
    """The customer's intent for their latest message."""

    intent: Literal["requirement_inquiry", "refund_request"] = Field(
        description="refund_request if the customer wants money back for a recharge "
                    "(or is answering the assistant's refund question); otherwise requirement_inquiry."
    )


SYSTEM = """You classify messages sent to NovaCell's telecom support chatbot.

Return refund_request when the customer asks NovaCell to refund, cancel or reverse a recharge \
or payment, i.e. they want the action done for them. Examples:
- "I want a refund of ₹999." -> refund_request
- "Please reverse my last recharge" -> refund_request
- "Cancel my ₹499 recharge and give my money back" -> refund_request
- "I want a refund" -> refund_request

Return requirement_inquiry for every other question, including questions about HOW refunds, \
cancellations or billing work. Examples:
- "Why is my internet slow?" -> requirement_inquiry
- "How do I cancel a recharge?" -> requirement_inquiry
- "What is your refund policy?" -> requirement_inquiry
- "Why is my bill higher than usual?" -> requirement_inquiry

{pending}"""

PENDING_NOTE = """Right now the assistant is waiting for the customer's reply about a refund: {what}. \
Short replies that answer it (an amount, "the smaller one", "yes", "no", "maybe", "not sure") \
are refund_request. A new, unrelated support question is still requirement_inquiry."""


def classify_intent(llm, message: str, history: list[dict], pending: str | None = None) -> str:
    """Return REQUIREMENT or REFUND. Raises on LLM failure; callers decide the fallback."""
    msgs = [SystemMessage(SYSTEM.format(pending=PENDING_NOTE.format(what=pending) if pending else ""))]
    for m in history[-4:]:  # a little context resolves short follow-ups
        msgs.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))
    msgs.append(HumanMessage(message))
    result = llm.with_structured_output(IntentResult).invoke(msgs)
    return result.intent
