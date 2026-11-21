"""Streamlit chat UI: support questions (RAG) and refund requests (Refund Agent)."""
import streamlit as st

from account_data import PRESETS
from assistant import SupportAssistant
from rag_chain import Timer, log_feedback, log_interaction, source_list
from refund_tools import fmt_inr

st.set_page_config(page_title="NovaCell Support Assistant", page_icon="📶")

QUESTION_SAMPLES = [
    "Why is my mobile internet so slow?",
    "How do I activate international roaming before travelling?",
    "My SIM card is not recognised after a phone update",
    "My eSIM activation keeps failing",
    "How do I download an itemised bill?",
    "I hear an echo on every call",
]
REFUND_SAMPLES = [
    "I want a refund of ₹499",
    "I want a refund of ₹999",
    "I want a refund",
    "Give me a refund of ₹2,000",
]
CONFIRM_TEXT, CANCEL_TEXT = "Yes, proceed", "Cancel"
SOURCE_ICON = {"FAQ": "❓", "TICKET": "🎫", "GUIDE": "📘"}

st.title("NovaCell Support Assistant")
st.caption("Ask about data, billing, SIM/eSIM, roaming, calls or devices, or request a refund "
           "for a recent recharge.")


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_assistant():
    return SupportAssistant()


try:
    assistant = get_assistant()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if "chat" not in st.session_state:
    st.session_state.chat = assistant.new_session()
    st.session_state.messages = []
chat = st.session_state.chat


def ask(text):
    st.session_state.pending = text


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("📶 NovaCell Support")
    st.caption("Answers come only from NovaCell's FAQ, resolved tickets and guides.")
    st.subheader("Ask a question")
    for q in QUESTION_SAMPLES:
        st.button(q, key=f"sample-{q}", width="stretch", on_click=ask, args=(q,))
    st.subheader("Try a refund")
    for q in REFUND_SAMPLES:
        st.button(q, key=f"sample-{q}", width="stretch", on_click=ask, args=(q,))
    st.divider()
    if st.button("Clear conversation", icon=":material/delete:", width="stretch"):
        chat.clear()
        st.session_state.messages = []
        st.session_state.pop("pending", None)
        st.rerun()

    with st.expander("Demo account", icon=":material/person:"):
        account = chat.agent.tools.get_user_account(chat.agent.user_id)
        st.caption(f"Signed in as **{account['name']}** ({account['user_id']}), "
                   f"{account['plan']} plan. Simulated data; no real payments.")
        st.dataframe(
            [{"Recharge": r["recharge_id"], "Amount": fmt_inr(r["recharge_amount"]),
              "Age (days)": r["recharge_age_days"], "Status": r["recharge_status"],
              "Refund": r["refund_status"]} for r in account["recharges"]],
            hide_index=True,
        )
        for ref in chat.store.refunds:
            st.caption(f"Refunded {fmt_inr(ref['amount'])} ({ref['refund_id']})")
        for req in chat.store.support_requests:
            st.caption(f"Sent to Support Team: {fmt_inr(req['amount'])} ({req['request_id']})")
        preset = st.selectbox("Scenario data", list(PRESETS), key="preset",
                              format_func=lambda k: PRESETS[k][0])
        reset = st.button("Reset demo account", icon=":material/restart_alt:")
        if reset or preset != chat.store.preset:
            chat.reset_account(preset)
            st.rerun()
    st.caption("No account access for questions; refunds use the simulated demo account above.")


def render_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"{SOURCE_ICON.get(s['source'], '•')} **{s['source']} · {s['ref']}** — {s['title']}")
            snippet = s["snippet"]
            st.caption(snippet[:400] + ("…" if len(snippet) > 400 else ""))


def render_feedback(idx, msg):
    if not msg.get("log_id"):
        return
    rating = msg.get("rating")
    if rating:
        st.caption("Thanks for the feedback! " + ("👍" if rating == "up" else "👎"))
        return
    with st.container(horizontal=True):
        for value, label in (("up", "👍"), ("down", "👎")):
            if st.button(label, key=f"fb-{value}-{idx}"):
                log_feedback(msg["log_id"], value)
                msg["rating"] = value
                st.rerun()


def render_confirmation():
    """Confirm / Cancel step for a pending refund (PRD section 11)."""
    state = chat.agent.state
    label = "Confirm refund" if state.action == "refund" else "Submit to Support Team"
    with st.container(horizontal=True):
        st.button(label, type="primary", icon=":material/check:", key="confirm-refund",
                  on_click=ask, args=(CONFIRM_TEXT,))
        st.button("Cancel", icon=":material/close:", key="cancel-refund",
                  on_click=ask, args=(CANCEL_TEXT,))


# ---------------- history ----------------
messages = st.session_state.messages
awaiting = chat.agent.state.status == "awaiting_confirmation"
for i, msg in enumerate(messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] != "assistant":
            continue
        if msg.get("note"):
            st.info(msg["note"], icon=":material/pending:")
        if msg.get("route") == "rag":
            render_sources(msg["sources"])
        if msg.get("trace"):
            with st.expander("Agent steps", icon=":material/account_tree:", type="compact"):
                st.code("\n".join(msg["trace"]), language=None)
        if awaiting and i == len(messages) - 1:
            render_confirmation()
        render_feedback(i, msg)

# ---------------- new turn ----------------
question = st.chat_input("Ask a question or request a refund...") or st.session_state.pop("pending", None)
if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        timer, first = Timer(), {}
        reply = None
        try:
            with st.spinner("Thinking..."):
                reply = assistant.handle(chat, question)
            if reply.route == "rag":
                def tokens():
                    for t in reply.stream:
                        first.setdefault("t", timer.elapsed())
                        yield t
                answer = st.write_stream(tokens())
            else:
                first["t"] = timer.elapsed()
                answer = reply.text
                st.markdown(answer)
            chat.add_exchange(question, answer)
        except Exception as e:  # network / API errors
            answer = ("Sorry, I couldn't reach the answer service right now. "
                      "Please try again, or call 611 / use the MyTelecom app.")
            st.error(f"{answer}\n\n`{type(e).__name__}: {e}`")
    docs = reply.docs if reply else []
    log_id = log_interaction(question, answer, docs, timer.elapsed(), first.get("t", 0.0), "web",
                             reply.intent if reply else None, reply.trace if reply else None)
    messages.append({
        "role": "assistant", "content": answer, "log_id": log_id,
        "route": reply.route if reply else None,
        "sources": source_list(docs),
        "trace": list(reply.trace) if reply else [],
        "note": reply.note if reply else None,
    })
    st.rerun()
