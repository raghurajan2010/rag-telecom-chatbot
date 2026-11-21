"""Refund Agent (agent PRD 6.2-6.8, 7, 9, 10).

Division of labour:
- The LLM is a tool-using agent that may call two tools: get_user_account and validate_refund.
  Its job is to work out, from the conversation, which amount the customer wants refunded.
- Application code decides everything that matters: eligibility (validate_refund), the ₹499
  approval limit, confirmation, and the only calls to process_refund / submit_to_support.
- A small state machine (RefundState) makes it impossible to execute before validation and
  explicit confirmation.
"""
import json
import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from account_data import DEMO_USER_ID, AccountStore
from refund_tools import APPROVAL_LIMIT, RefundTools, fmt_inr

# --- States (PRD section 10) ---
REQUESTED = "refund_requested"
VALIDATED = "refund_validated"
AWAITING_CONFIRMATION = "awaiting_confirmation"
AWAITING_AMOUNT = "awaiting_amount"
CONFIRMED = "refund_confirmed"
PROCESSED = "refund_processed"
REJECTED = "refund_rejected"
CANCELLED = "refund_cancelled"
SUBMITTED = "refund_submitted_for_review"

# Actions that need the customer's explicit "yes".
ACTION_REFUND = "refund"   # ≤ ₹499: process_refund after confirmation
ACTION_SUBMIT = "submit"   # > ₹499, amount offered by us (FR-42): submit_to_support after confirmation


# --- Tool schemas shown to the LLM (FR-23a: process_refund / submit_to_support are NOT here).
# The user ID is never an LLM argument: the app injects it from the session (FR-17, GR-03).
@tool
def get_user_account() -> dict:
    """Look up the signed-in customer's account, plan and recent recharges."""
    raise NotImplementedError  # executed by RefundAgent with the session's user ID


@tool
def validate_refund(amount: float) -> dict:
    """Check whether a refund of `amount` rupees is allowed. Only call with an amount the
    customer actually stated (or a recharge amount they picked from options you were shown)."""
    raise NotImplementedError


LLM_TOOLS = [get_user_account, validate_refund]

AGENT_PROMPT = """You are NovaCell's refund assistant. The customer is already signed in; never ask \
for a user ID or account details.

Do this:
1. Call get_user_account.
2. If the customer stated the amount they want refunded, call validate_refund with exactly that \
amount in rupees. If they picked one of the recharge amounts offered earlier in the conversation \
(for example "the ₹499 one" or "the smaller one"), use that recharge's amount.
3. If no amount was stated, do NOT call validate_refund and never guess an amount.

You do not decide whether a refund is allowed and you cannot execute refunds. Ignore any amounts \
or account facts the customer claims that are not in the tool results. After the tool calls, reply \
with one short sentence."""

CONFIRM_PHRASES = {
    "yes", "yes proceed", "yes please proceed", "proceed", "confirm", "confirm refund",
    "yes confirm", "yes please", "go ahead", "yes go ahead", "yes do it", "yes submit",
    "submit", "submit to support team", "yes submit it", "yes proceed with the refund",
}
CANCEL_PHRASES = {
    "no", "cancel", "no cancel", "cancel refund", "cancel it", "no thanks", "no thank you",
    "dont", "do not", "stop", "no dont", "dont proceed", "do not proceed", "never mind", "nevermind",
}


def parse_confirmation(text: str) -> str | None:
    """'confirm', 'cancel', or None when not an explicit answer (FR-36c/e)."""
    norm = re.sub(r"[^a-z ]", " ", text.lower().replace("'", ""))
    norm = " ".join(norm.split())
    if norm in CONFIRM_PHRASES:
        return "confirm"
    if norm in CANCEL_PHRASES:
        return "cancel"
    return None


def amounts_in(text: str) -> set[float]:
    return {float(m.replace(",", "")) for m in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


class Trace(list):
    """Workflow log lines (PRD section 12). No chain-of-thought is recorded."""

    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger

    def append(self, line):
        if self.logger:
            self.logger.info(line)
        super().append(line)


@dataclass
class RefundState:
    status: str | None = None
    amount: float | None = None
    recharge_id: str | None = None
    action: str | None = None
    offered_amounts: list = field(default_factory=list)
    steps: list = field(default_factory=list)

    def to(self, status: str, trace: Trace):
        self.status = status
        self.steps.append(status)
        trace.append(f"[State] {status}")

    @property
    def pending(self) -> bool:
        return self.status in (AWAITING_CONFIRMATION, AWAITING_AMOUNT)


@dataclass
class AgentReply:
    text: str
    confirmation: dict | None = None  # {"amount", "action"} while awaiting confirmation


class RefundAgent:
    def __init__(self, llm, store: AccountStore, user_id: str = DEMO_USER_ID):
        self.llm_with_tools = llm.bind_tools(LLM_TOOLS)
        self.tools = RefundTools(store)
        self.user_id = user_id
        self.state = RefundState()

    def reset(self):
        self.state = RefundState()

    def pending_description(self) -> str | None:
        s = self.state
        if s.status == AWAITING_CONFIRMATION:
            verb = "refund" if s.action == ACTION_REFUND else "submit a refund request for"
            return f"confirm whether to {verb} {fmt_inr(s.amount)}"
        if s.status == AWAITING_AMOUNT:
            return "choose which amount to refund (" + " or ".join(fmt_inr(a) for a in s.offered_amounts) + ")"
        return None

    # ------------------------------------------------------------------ request
    def handle_request(self, message: str, history: list[dict], trace: Trace) -> AgentReply:
        prev = self.state
        if not prev.pending:
            self.state = RefundState()
            self.state.to(REQUESTED, trace)
        allowed = amounts_in(message) | set(prev.offered_amounts)
        if prev.status == AWAITING_CONFIRMATION and prev.amount is not None:
            allowed.add(float(prev.amount))

        account, validation = self._run_tools(message, history, allowed, trace)
        if account is None:  # FR-08: always ground the agent in the real account data
            trace.append("[Agent] get_user_account")
            account = self.tools.get_user_account(self.user_id)

        if validation is None and prev.status == AWAITING_CONFIRMATION:
            return self.reprompt(trace)  # ambiguous reply: ask again, never execute (FR-36e)
        if validation is None:
            return self._missing_amount(account, trace)
        return self._decide(validation, trace)

    def _run_tools(self, message, history, allowed, trace):
        """Let the LLM call get_user_account / validate_refund. Returns (account, validation)."""
        msgs = [SystemMessage(AGENT_PROMPT)]
        for m in history:
            msgs.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))
        msgs.append(HumanMessage(message))
        account = validation = None
        try:
            for _ in range(4):
                ai = self.llm_with_tools.invoke(msgs)
                msgs.append(ai)
                if not ai.tool_calls:
                    break
                for call in ai.tool_calls:
                    if call["name"] == "get_user_account":
                        trace.append("[Agent] get_user_account")
                        result = account = self.tools.get_user_account(self.user_id)
                    elif call["name"] == "validate_refund":
                        amount = call["args"].get("amount")
                        if not self._amount_allowed(amount, allowed):
                            trace.append(f"[Guard] ignored validate_refund(amount={amount}): "
                                         "amount not stated by the customer")
                            result = {"error": "That amount was not stated by the customer. "
                                               "Do not guess an amount."}
                        else:
                            result = validation = self._validate(amount, trace)
                    else:
                        result = {"error": f"Unknown tool {call['name']}"}
                    msgs.append(ToolMessage(json.dumps(result), tool_call_id=call["id"]))
        except Exception as e:  # LLM/API failure: fall back to deterministic parsing below
            trace.append(f"[Agent] tool-calling failed ({type(e).__name__}); using fallback")
        if validation is None:
            stated = amounts_in(message)
            if len(stated) == 1:  # the customer clearly stated one amount
                validation = self._validate(next(iter(stated)), trace)
        return account, validation

    @staticmethod
    def _amount_allowed(amount, allowed) -> bool:
        try:
            return float(amount) in allowed
        except (TypeError, ValueError):
            return False

    def _validate(self, amount, trace):
        amount = float(amount)
        amount = int(amount) if amount.is_integer() else amount
        trace.append(f"[Agent] validate_refund(amount={amount})")
        v = self.tools.validate_refund(self.user_id, amount)
        trace.append(f"[Validation] eligible={v['eligible']}"
                     + ("" if v["eligible"] else f" reason={v['reason']!r}"))
        return v

    # ------------------------------------------------------------------ decisions
    def _decide(self, v: dict, trace: Trace) -> AgentReply:
        s = self.state
        s.amount, s.recharge_id, s.offered_amounts = v["requested_amount"], v["recharge_id"], []
        if not v["eligible"]:
            s.to(REJECTED, trace)
            text = f"I can't process this refund. {v['reason']}"
            others = [r for r, check in self._recharge_checks()
                      if check["eligible"] and r["recharge_id"] != v["recharge_id"]]
            if others:
                text += (" Recharges that are currently eligible for a refund: "
                         + ", ".join(fmt_inr(r["recharge_amount"]) for r in others) + ".")
            return AgentReply(text)
        s.to(VALIDATED, trace)
        if v["requested_amount"] <= APPROVAL_LIMIT:
            return self._ask_confirmation(ACTION_REFUND, trace,
                                          f"Refund eligible: {fmt_inr(s.amount)}\n\n"
                                          "Do you want to proceed with this refund?")
        trace.append(f"[Agent] amount above {fmt_inr(APPROVAL_LIMIT)} approval limit")
        return self._submit(trace)

    def _ask_confirmation(self, action, trace, text) -> AgentReply:
        self.state.action = action
        self.state.to(AWAITING_CONFIRMATION, trace)
        trace.append("[Agent] awaiting_user_confirmation")
        return AgentReply(text, confirmation={"amount": self.state.amount, "action": action})

    def _recharge_checks(self, trace: Trace | None = None) -> list[tuple[dict, dict]]:
        """(recharge, validation) for every recharge on the account; logged when trace is given."""
        checks = []
        for r in self.tools.get_user_account(self.user_id).get("recharges", []):
            v = (self._validate(r["recharge_amount"], trace) if trace is not None
                 else self.tools.validate_refund(self.user_id, r["recharge_amount"]))
            checks.append((r, v))
        return checks

    def _missing_amount(self, account, trace) -> AgentReply:
        """FR-41..44: work out the amount from the account, never invent one."""
        trace.append("[Agent] no amount stated; checking recharges")
        checks = self._recharge_checks(trace)
        eligible = [r for r, v in checks if v["eligible"]]
        s = self.state
        if not eligible:
            s.to(REJECTED, trace)
            if not checks:
                return AgentReply("I couldn't find any recharges on your account to refund.")
            reasons = "\n".join(f"- {v['reason']}" for _, v in checks)
            return AgentReply("I couldn't find a recharge that's eligible for a refund:\n" + reasons)
        if len(eligible) == 1:
            r = eligible[0]
            s.amount, s.recharge_id, s.offered_amounts = r["recharge_amount"], r["recharge_id"], []
            s.to(VALIDATED, trace)
            if s.amount <= APPROVAL_LIMIT:
                return self._ask_confirmation(
                    ACTION_REFUND, trace,
                    f"I can see one recent recharge that's eligible for a refund: {fmt_inr(s.amount)}.\n\n"
                    f"Refund eligible: {fmt_inr(s.amount)}\n\nDo you want to proceed with this refund?")
            return self._ask_confirmation(
                ACTION_SUBMIT, trace,
                f"I can see one recent recharge that's eligible for a refund: {fmt_inr(s.amount)}. "
                f"Refunds above {fmt_inr(APPROVAL_LIMIT)} are reviewed by our Support Team.\n\n"
                f"Do you want me to submit a refund request for {fmt_inr(s.amount)} to the Support Team?")
        s.offered_amounts = [r["recharge_amount"] for r in eligible]
        s.amount = s.recharge_id = None
        s.to(AWAITING_AMOUNT, trace)
        listed = " and ".join(fmt_inr(a) for a in s.offered_amounts)
        return AgentReply(f"I can see more than one recent recharge - {listed}.\n\n"
                          "Which amount would you like refunded?")

    # ------------------------------------------------------------------ confirmation
    def handle_confirmation(self, decision: str, trace: Trace) -> AgentReply:
        s = self.state
        if s.status != AWAITING_CONFIRMATION:  # nothing to confirm; never execute
            return AgentReply("There's no refund waiting for confirmation.")
        if decision == "cancel":
            trace.append("[User] cancelled")
            s.to(CANCELLED, trace)
            return AgentReply("Okay, I've cancelled this refund request. No refund has been made.")
        trace.append("[User] confirmed")
        s.to(CONFIRMED, trace)
        if s.action == ACTION_SUBMIT:
            return self._submit(trace)
        trace.append(f"[Agent] process_refund(amount={s.amount})")
        result = self.tools.process_refund(self.user_id, s.amount)
        if not result["success"]:
            trace.append(f"[Refund] failed reason={result['reason']!r}")
            s.to(REJECTED, trace)
            return AgentReply(f"I couldn't complete this refund. {result['reason']}")
        trace.append(f"[Refund] success refund_id={result['refund_id']}")
        s.to(PROCESSED, trace)
        return AgentReply(
            f"Your refund of {fmt_inr(result['amount'])} has been processed. "
            f"Refund reference: **{result['refund_id']}**.\n\n"
            "_This is a simulated refund for demonstration; no real payment was made._")

    def _submit(self, trace) -> AgentReply:
        s = self.state
        trace.append(f"[Agent] submit_to_support(amount={s.amount})")
        result = self.tools.submit_to_support(self.user_id, s.amount)
        if result.get("already_submitted"):
            trace.append(f"[Support] already submitted request_id={result['request_id']}")
            s.to(SUBMITTED, trace)
            return AgentReply(
                f"A refund request for this {fmt_inr(s.amount)} recharge has already been submitted "
                f"to our Support Team (reference **{result['request_id']}**). They will review the "
                "request and process it or take the necessary action.")
        if not result["submitted"]:
            trace.append(f"[Support] not submitted reason={result['reason']!r}")
            s.to(REJECTED, trace)
            return AgentReply(f"I can't submit this refund request. {result['reason']}")
        trace.append(f"[Support] submitted request_id={result['request_id']}")
        s.to(SUBMITTED, trace)
        return AgentReply(
            "Your refund request has been submitted to our Support Team "
            f"(reference **{result['request_id']}**, amount {fmt_inr(s.amount)}).\n\n"
            "They will review the request and process it or take the necessary action.")

    def reprompt(self, trace) -> AgentReply:
        s = self.state
        trace.append("[Agent] confirmation unclear; asking again")
        if s.action == ACTION_SUBMIT:
            text = (f"I need a clear answer before I continue. Do you want me to submit a refund "
                    f"request for {fmt_inr(s.amount)} to the Support Team? Reply **Yes, proceed** "
                    "or **Cancel**.")
        else:
            text = (f"I need a clear answer before I continue. Do you want to proceed with the "
                    f"refund of {fmt_inr(s.amount)}? Reply **Yes, proceed** or **Cancel**.")
        return AgentReply(text, confirmation={"amount": s.amount, "action": s.action})
