"""Run the agent PRD's demonstration scenarios end to end against the live model.

    python demo_scenarios.py

Each scenario prints the conversation and the workflow log, then checks the expected outcome.
Exits with status 1 if any check fails.
"""
import sys

from assistant import SupportAssistant

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

FAILURES = []


def say(assistant, session, text):
    reply = assistant.handle(session, text)
    answer = reply.text if reply.route == "refund" else "".join(reply.stream)
    session.add_exchange(text, answer)
    print(f"\nUser: {text}\nBot [{reply.route}]: {answer}")
    if reply.note:
        print(f"  (note: {reply.note})")
    for line in reply.trace:
        print(f"    {line}")
    return reply, answer


def check(name, condition):
    print(f"  {'PASS' if condition else 'FAIL'}: {name}")
    if not condition:
        FAILURES.append(name)


def recharge(session, rid):
    return next(r for r in session.store.users["user_123"]["recharges"] if r["recharge_id"] == rid)


def called(reply, tool):
    return any(line.startswith(f"[Agent] {tool}") for line in reply.trace)


def main():
    a = SupportAssistant()

    print("\n=== Scenario 1: successful refund of ₹499 ===")
    s = a.new_session()
    r, _ = say(a, s, "I want a refund of ₹499.")
    check("routed to refund agent", r.route == "refund")
    check("asks for confirmation", r.confirmation is not None)
    check("not refunded before confirmation", recharge(s, "rch_457")["refund_status"] == "not_refunded")
    r, ans = say(a, s, "Yes, proceed")
    check("process_refund called after confirmation", called(r, "process_refund"))
    check("refund reference returned", "ref_" in ans)
    check("₹499 recharge marked refunded", recharge(s, "rch_457")["refund_status"] == "refunded")
    check("₹999 recharge still refundable", recharge(s, "rch_456")["refund_status"] == "not_refunded")

    print("\n=== Scenario 2: amount exceeds recharge ===")
    s = a.new_session()
    r, ans = say(a, s, "Give me a refund of ₹2,000.")
    check("rejected: exceeds recharge amount", "exceeds" in ans)
    check("process_refund not called", not called(r, "process_refund"))

    print("\n=== Scenario 3: recharge too old ===")
    s = a.new_session("old_recharge")
    r, ans = say(a, s, "I want a refund of ₹499.")
    check("rejected: outside 7-day window", "7-day" in ans)
    check("process_refund not called", not called(r, "process_refund"))

    print("\n=== Scenario 4: already refunded ===")
    s = a.new_session("already_refunded")
    r, ans = say(a, s, "I want a refund of ₹499.")
    check("rejected: already refunded", "already been refunded" in ans)
    check("process_refund not called", not called(r, "process_refund"))

    print("\n=== Scenario 5: existing RAG query ===")
    s = a.new_session()
    r, ans = say(a, s, "Why is my mobile internet slow?")
    check("routed to RAG", r.route == "rag" and len(r.docs) == 9)

    print("\n=== Scenario 6: refund above the approval limit ===")
    s = a.new_session()
    r, ans = say(a, s, "I want a refund of ₹999.")
    check("submitted to Support Team", called(r, "submit_to_support") and "Support Team" in ans)
    check("no confirmation step", r.confirmation is None)
    check("process_refund not called", not called(r, "process_refund"))
    check("refund status unchanged", recharge(s, "rch_456")["refund_status"] == "not_refunded")
    check("no claimed outcome", not any(w in ans.lower() for w in ("approved", "has been processed", "rejected")))

    print("\n=== Scenario 7: follow-up question uses history ===")
    s = a.new_session()
    say(a, s, "How do I activate international roaming?")
    r, ans = say(a, s, "Should I do that before I travel?")
    check("routed to RAG", r.route == "rag")
    check("answer is about roaming before travel", "roaming" in ans.lower() or "24 hours" in ans)

    print("\n=== Extra: missing amount, then choose, then cancel ===")
    s = a.new_session()
    r, ans = say(a, s, "I want a refund.")
    check("asks which amount (₹999 and ₹499)", "₹999" in ans and "₹499" in ans and r.confirmation is None)
    r, ans = say(a, s, "the ₹499 one")
    check("then asks confirmation for ₹499", r.confirmation and r.confirmation["amount"] == 499)
    r, ans = say(a, s, "hmm maybe")
    check("ambiguous reply asks again, no refund", r.confirmation is not None
          and recharge(s, "rch_457")["refund_status"] == "not_refunded")
    r, ans = say(a, s, "Cancel")
    check("cancelled, nothing refunded", "cancelled" in ans and
          recharge(s, "rch_457")["refund_status"] == "not_refunded")

    print("\n=== Extra: question about how to cancel is not a refund request ===")
    s = a.new_session()
    r, _ = say(a, s, "How do I cancel a recharge?")
    check("routed to RAG", r.route == "rag")

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: {FAILURES}'}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
