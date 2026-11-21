"""Offline tests for the deterministic refund rules and guardrails (no API key or network needed).

    python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from account_data import DEMO_USER_ID, AccountStore  # noqa: E402
from refund_agent import (  # noqa: E402
    AWAITING_AMOUNT, AWAITING_CONFIRMATION, CANCELLED, PROCESSED, REJECTED, SUBMITTED,
    RefundAgent, Trace, parse_confirmation,
)
from refund_tools import RefundTools  # noqa: E402

U = DEMO_USER_ID


class FakeLLM:
    """Stands in for the chat model: never calls tools, so the agent's own fallback logic runs."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        return AIMessage(content="ok")


def recharge(store, rid):
    return next(r for r in store.users[U]["recharges"] if r["recharge_id"] == rid)


class ValidationRules(unittest.TestCase):
    def setUp(self):
        self.store = AccountStore()
        self.tools = RefundTools(self.store)

    def test_eligible_499_selects_rch_457(self):
        v = self.tools.validate_refund(U, 499)
        self.assertEqual((v["eligible"], v["recharge_id"], v["recharge_amount"]), (True, "rch_457", 499))

    def test_amount_must_be_positive(self):
        self.assertFalse(self.tools.validate_refund(U, 0)["eligible"])
        self.assertFalse(self.tools.validate_refund(U, -5)["eligible"])

    def test_amount_cannot_exceed_recharge(self):
        v = self.tools.validate_refund(U, 2000)
        self.assertFalse(v["eligible"])
        self.assertIn("exceeds", v["reason"])

    def test_outside_window(self):
        tools = RefundTools(AccountStore("old_recharge"))
        v = tools.validate_refund(U, 499)
        self.assertFalse(v["eligible"])
        self.assertIn("7-day", v["reason"])

    def test_already_refunded_exact_match_still_selected(self):
        tools = RefundTools(AccountStore("already_refunded"))
        v = tools.validate_refund(U, 499)
        self.assertEqual((v["eligible"], v["recharge_id"]), (False, "rch_457"))

    def test_not_completed(self):
        recharge(self.store, "rch_457")["recharge_status"] = "pending"
        self.assertFalse(self.tools.validate_refund(U, 499)["eligible"])

    def test_partial_amount_uses_covering_recharge(self):
        self.assertEqual(self.tools.validate_refund(U, 300)["recharge_id"], "rch_457")
        self.assertEqual(self.tools.validate_refund(U, 700)["recharge_id"], "rch_456")


class Execution(unittest.TestCase):
    def setUp(self):
        self.store = AccountStore()
        self.tools = RefundTools(self.store)

    def test_process_refund_updates_state_and_returns_reference(self):
        r = self.tools.process_refund(U, 499)
        self.assertTrue(r["success"])
        self.assertTrue(r["refund_id"].startswith("ref_"))
        self.assertEqual(recharge(self.store, "rch_457")["refund_status"], "refunded")
        self.assertEqual(recharge(self.store, "rch_456")["refund_status"], "not_refunded")

    def test_cannot_refund_twice(self):
        self.tools.process_refund(U, 499)
        self.assertFalse(self.tools.process_refund(U, 499)["success"])

    def test_process_refund_refuses_above_limit(self):
        r = self.tools.process_refund(U, 999)
        self.assertFalse(r["success"])
        self.assertEqual(recharge(self.store, "rch_456")["refund_status"], "not_refunded")

    def test_submit_to_support_leaves_status_unchanged(self):
        r = self.tools.submit_to_support(U, 999)
        self.assertTrue(r["submitted"])
        self.assertEqual(recharge(self.store, "rch_456")["refund_status"], "not_refunded")
        self.assertTrue(self.tools.submit_to_support(U, 999).get("already_submitted"))


class AgentStateMachine(unittest.TestCase):
    def setUp(self):
        self.store = AccountStore()
        self.agent = RefundAgent(FakeLLM(), self.store)

    def ask(self, text):
        return self.agent.handle_request(text, [], Trace())

    def test_confirmation_required_before_refund(self):
        reply = self.ask("refund ₹499 please")
        self.assertEqual(self.agent.state.status, AWAITING_CONFIRMATION)
        self.assertIsNotNone(reply.confirmation)
        self.assertEqual(recharge(self.store, "rch_457")["refund_status"], "not_refunded")
        self.agent.handle_confirmation("confirm", Trace())
        self.assertEqual(self.agent.state.status, PROCESSED)
        self.assertEqual(recharge(self.store, "rch_457")["refund_status"], "refunded")

    def test_cancel_executes_nothing(self):
        self.ask("refund ₹499 please")
        self.agent.handle_confirmation("cancel", Trace())
        self.assertEqual(self.agent.state.status, CANCELLED)
        self.assertEqual(self.store.refunds, [])

    def test_confirmation_without_pending_refund_does_nothing(self):
        self.agent.handle_confirmation("confirm", Trace())
        self.assertEqual(self.store.refunds, [])

    def test_ambiguous_reply_asks_again(self):
        self.ask("refund ₹499 please")
        reply = self.ask("hmm not sure")
        self.assertEqual(self.agent.state.status, AWAITING_CONFIRMATION)
        self.assertIsNotNone(reply.confirmation)
        self.assertEqual(self.store.refunds, [])

    def test_above_limit_is_submitted_without_confirmation(self):
        reply = self.ask("refund ₹999")
        self.assertEqual(self.agent.state.status, SUBMITTED)
        self.assertIsNone(reply.confirmation)
        self.assertEqual(self.store.refunds, [])
        self.assertNotIn("approved", reply.text.lower())

    def test_missing_amount_with_two_eligible_asks_which(self):
        reply = self.ask("I want a refund")
        self.assertEqual(self.agent.state.status, AWAITING_AMOUNT)
        self.assertIn("₹999", reply.text)
        self.assertIn("₹499", reply.text)

    def test_missing_amount_with_one_eligible_offers_it(self):
        self.agent.tools.process_refund(U, 499)
        reply = self.ask("I want a refund")
        self.assertEqual(self.agent.state.status, AWAITING_CONFIRMATION)
        self.assertEqual(reply.confirmation["amount"], 999)
        self.assertEqual(reply.confirmation["action"], "submit")

    def test_invalid_refund_rejected(self):
        self.ask("refund ₹2000")
        self.assertEqual(self.agent.state.status, REJECTED)
        self.assertEqual(self.store.refunds, [])


class ConfirmationParsing(unittest.TestCase):
    def test_explicit(self):
        for text in ["Yes, proceed", "yes", "Confirm refund", "Go ahead!"]:
            self.assertEqual(parse_confirmation(text), "confirm", text)
        for text in ["Cancel", "No", "no thanks", "Don't"]:
            self.assertEqual(parse_confirmation(text), "cancel", text)

    def test_ambiguous(self):
        for text in ["maybe", "ok I guess", "I want a refund of 499", "yes but what about 999?"]:
            self.assertIsNone(parse_confirmation(text), text)


if __name__ == "__main__":
    unittest.main()
