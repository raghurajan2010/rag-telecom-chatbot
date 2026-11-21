"""Hard-coded demo account data (agent PRD 6.3). No real CRM or billing system.

Each chat session gets its own copy (AccountStore), so a simulated refund in one
browser tab doesn't change another tab's data.
"""
import copy

# Treated as if it came from the authenticated session (FR-15/16). Never asked from the user.
DEMO_USER_ID = "user_123"

USERS = {
    "user_123": {
        "name": "Rahul",
        "plan": "Pro",
        "plan_amount": 999,
        "recharges": [
            {
                "recharge_id": "rch_456",
                "recharge_amount": 999,
                "recharge_status": "completed",
                "recharge_age_days": 2,
                "refund_status": "not_refunded",
            },
            {
                "recharge_id": "rch_457",
                "recharge_amount": 499,
                "recharge_status": "completed",
                "recharge_age_days": 1,
                "refund_status": "not_refunded",
            },
        ],
    }
}

# Demo presets so every PRD scenario can be shown in the UI (Scenarios 3 and 4 need altered data).
PRESETS = {
    "default": ("Default: ₹999 (2 days) and ₹499 (1 day)", {}),
    "old_recharge": ("₹499 recharge is 15 days old (Scenario 3)",
                     {"rch_457": {"recharge_age_days": 15}}),
    "already_refunded": ("₹499 recharge already refunded (Scenario 4)",
                         {"rch_457": {"refund_status": "refunded"}}),
}


class AccountStore:
    """Per-session, in-memory copy of USERS plus simulated refunds and support requests."""

    def __init__(self, preset: str = "default"):
        self.preset = preset
        self.users = copy.deepcopy(USERS)
        for recharge_id, changes in PRESETS[preset][1].items():
            for user in self.users.values():
                for r in user["recharges"]:
                    if r["recharge_id"] == recharge_id:
                        r.update(changes)
        self.refunds: list[dict] = []           # simulated refunds executed
        self.support_requests: list[dict] = []  # requests handed to the Support Team
