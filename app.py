"""Streamlit chat UI for the telecom RAG chatbot."""
import streamlit as st

from rag_chain import TelecomRAG, Timer, log_feedback, log_interaction, source_list

st.set_page_config(page_title="NovaCell Support Assistant", page_icon="📶")

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "Why was I charged for roaming after my trip?",
    "How do I activate international roaming before travelling?",
    "My SIM card is not recognised after a phone update",
    "My eSIM activation keeps failing",
    "I was double charged for my monthly plan",
    "How do I download an itemised bill?",
    "I hear an echo on every call",
    "I can't log in to the MyTelecom app",
]
SOURCE_ICON = {"FAQ": "❓", "TICKET": "🎫", "GUIDE": "📘"}


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_bot():
    return TelecomRAG()


if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("📶 NovaCell Support")
    st.caption("Answers come only from NovaCell's FAQ, resolved tickets and guides.")
    st.subheader("Try a sample question")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("pending", None)
        st.rerun()
    st.caption("No account access — for balances or personal bills use the MyTelecom app or call 611.")


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
    c1, c2, _ = st.columns([1, 1, 10])
    for col, value, label in ((c1, "up", "👍"), (c2, "down", "👎")):
        if col.button(label, key=f"fb-{value}-{idx}"):
            log_feedback(msg["log_id"], value)
            msg["rating"] = value
            st.rerun()


st.title("NovaCell Support Assistant")
st.caption("Ask about data, billing, SIM/eSIM, roaming, calls, devices or your account.")

try:
    bot = get_bot()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# ---------------- history ----------------
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg["sources"])
            render_feedback(i, msg)

# ---------------- new turn ----------------
question = st.chat_input("Type your question...") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        timer = Timer()
        docs = bot.retrieve(question)
        first = {}

        def tokens():
            for t in bot.stream_answer(question, docs):
                first.setdefault("t", timer.elapsed())
                yield t

        try:
            answer = st.write_stream(tokens())
        except Exception as e:  # network / API errors
            answer = ("Sorry, I couldn't reach the answer service right now. "
                      "Please try again, or call 611 / use the MyTelecom app.")
            st.error(f"{answer}\n\n`{type(e).__name__}: {e}`")
        log_id = log_interaction(question, answer, docs, timer.elapsed(), first.get("t", 0.0), "web")
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": source_list(docs), "log_id": log_id})
    st.rerun()
