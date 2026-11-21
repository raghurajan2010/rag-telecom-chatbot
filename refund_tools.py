"""Refund tools (agent PRD 6.4-6.7). All eligibility rules are deterministic code (FR-29/30).

Only get_user_account and validate_refund are offered to the LLM (see refund_agent.py).
process_refund and submit_to_support are called by application code alone (FR-23a).
Nothing here touches a real payment or billing system (FR-35, GR-07).
"""
import uuid

from account_data import AccountStore

APPROVAL_LIMIT = 499        # ₹; refunds above this go to the Support Team (FR-36..40)
REFUND_WINDOW_DAYS = 7      # FR-25


def fmt_inr(amount) -> str:
    amount = float(amount)
    return f"₹{amount:,.0f}" if amount.is_integer() else f"₹{amount:,.2f}"


def _refundable(r: dict) -> bool:
    return (r["recharge_status"] == "completed"
            and r["recharge_age_days"] <= REFUND_WINDOW_DAYS
            and r["refund_status"] != "refunded")


class RefundTools:
    def __init__(self, store: AccountStore):
        self.store = store

    def _user(self, user_id: str):
        return self.store.users.get(user_id)

    # ---------------- FR-19 ----------------
    def get_user_account(self, user_id: str) -> dict:
        user = self._user(user_id)
        if user is None:
            return {"found": False, "user_id": user_id}
        return {
            "found": True,
            "user_id": user_id,
            "name": user["name"],
            "plan": user["plan"],
            "plan_amount": user["plan_amount"],
            "recharges": [dict(r) for r in user["recharges"]],
        }

    def select_recharge(self, user_id: str, amount) -> dict | None:
        """Pick which recharge a refund applies to, deterministically (FR-27a).

        1. A recharge whose amount matches exactly (even if already refunded); most recent first.
        2. Otherwise the most recent refundable recharge that covers the amount.
        3. Otherwise the most recent recharge (validation then explains why it fails).
        """
        user = self._user(user_id)
        if not user or not user["recharges"]:
            return None
        recharges = user["recharges"]
        by_recency = lambda r: (r["recharge_age_days"], r["recharge_amount"])  # noqa: E731
        exact = [r for r in recharges if float(r["recharge_amount"]) == float(amount)]
        if exact:
            return min(exact, key=by_recency)
        covering = [r for r in recharges if _refundable(r) and r["recharge_amount"] >= amount]
        if covering:
            return min(covering, key=by_recency)
        return min(recharges, key=by_recency)

    # ---------------- FR-20, FR-24..30 ----------------
    def validate_refund(self, user_id: str, amount) -> dict:
        def result(eligible, reason, recharge=None):
            return {
                "eligible": eligible,
                "reason": reason,
                "requested_amount": amount,
                "recharge_amount": recharge["recharge_amount"] if recharge else None,
                "recharge_id": recharge["recharge_id"] if recharge else None,
            }

        try:
            amount = float(amount)
            amount = int(amount) if amount.is_integer() else amount
        except (TypeError, ValueError):
            return result(False, "The refund amount is not a valid number.")
        if amount <= 0:
            return result(False, "The refund amount must be greater than zero.")
        if self._user(user_id) is None:
            return result(False, "No account was found for this session.")
        r = self.select_recharge(user_id, amount)
        if r is None:
            return result(False, "There are no recharges on this account.")
        if r["recharge_status"] != "completed":
            return result(False, f"Recharge {r['recharge_id']} is not completed "
                                 f"(status: {r['recharge_status']}).", r)
        if r["recharge_age_days"] > REFUND_WINDOW_DAYS:
            return result(False, f"The {fmt_inr(r['recharge_amount'])} recharge was made "
                                 f"{r['recharge_age_days']} days ago, outside the "
                                 f"{REFUND_WINDOW_DAYS}-day refund window.", r)
        if r["refund_status"] == "refunded":
            return result(False, f"The {fmt_inr(r['recharge_amount'])} recharge "
                                 f"({r['recharge_id']}) has already been refunded.", r)
        if amount > r["recharge_amount"]:
            return result(False, f"The requested amount of {fmt_inr(amount)} exceeds the "
                                 f"recharge amount of {fmt_inr(r['recharge_amount'])}.", r)
        return result(True, "Refund is eligible.", r)

    # ---------------- FR-21, FR-31..35 (app code only) ----------------
    def process_refund(self, user_id: str, amount) -> dict:
        """Simulated refund. Re-validates and re-checks the approval limit (GR-01, GR-08)."""
        v = self.validate_refund(user_id, amount)
        if not v["eligible"]:
            return {"success": False, "reason": v["reason"], "amount": amount}
        if v["requested_amount"] > APPROVAL_LIMIT:
            return {"success": False, "amount": amount,
                    "reason": f"Refunds above {fmt_inr(APPROVAL_LIMIT)} can't be processed here."}
        recharge = self.select_recharge(user_id, v["requested_amount"])
        recharge["refund_status"] = "refunded"  # FR-28a: the whole recharge is consumed
        refund = {
            "success": True,
            "refund_id": f"ref_{uuid.uuid4().hex[:6]}",
            "amount": v["requested_amount"],
            "recharge_id": recharge["recharge_id"],
            "status": "refunded",
        }
        self.store.refunds.append(refund)
        return refund

    # ---------------- FR-21a, FR-38..40 (app code only) ----------------
    def submit_to_support(self, user_id: str, amount) -> dict:
        """Hand an eligible above-limit refund to the Support Team. Refund status is unchanged."""
        v = self.validate_refund(user_id, amount)
        if not v["eligible"]:
            return {"submitted": False, "reason": v["reason"], "amount": amount}
        existing = next((s for s in self.store.support_requests
                         if s["recharge_id"] == v["recharge_id"]), None)
        if existing:
            return {"submitted": False, "already_submitted": True, **existing}
        request = {
            "request_id": f"sup_{uuid.uuid4().hex[:6]}",
            "amount": v["requested_amount"],
            "recharge_id": v["recharge_id"],
            "status": "submitted_for_review",
        }
        self.store.support_requests.append(request)
        return {"submitted": True, **request}
