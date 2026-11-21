"""CLI REPL for the telecom support assistant (questions + refunds). Type 'quit' to exit."""
import sys

from assistant import SupportAssistant
from rag_chain import Timer, log_interaction


def main():
    # Avoid crashes on Windows consoles that can't encode emoji/unicode in model output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print("Loading knowledge base and models...")
    assistant = SupportAssistant()
    session = assistant.new_session()
    print("NovaCell Telecom Assistant - ask a question or request a refund.")
    print("Commands: 'clear' resets the conversation, 'quit' exits.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "clear":
            session.clear()
            print("Conversation cleared.\n")
            continue
        timer = Timer()
        reply = assistant.handle(session, question)
        print("Bot: ", end="", flush=True)
        first = None
        if reply.route == "rag":
            answer = ""
            for token in reply.stream:
                first = first if first is not None else timer.elapsed()
                answer += token
                sys.stdout.write(token)
                sys.stdout.flush()
            print("\n\nSources:")
            for d in reply.docs:
                print(f"  - {d.metadata.get('source')}: {d.metadata.get('ref')}")
            if reply.note:
                print(f"\n({reply.note})")
        else:
            answer = reply.text
            first = timer.elapsed()
            print(answer.replace("**", ""))
        if reply.confirmation:
            print("\nType 'Yes, proceed' to confirm or 'Cancel' to stop.")
        print()
        session.add_exchange(question, answer)
        log_interaction(question, answer, reply.docs, timer.elapsed(), first or 0.0, "cli",
                        reply.intent, reply.trace)
    print("Goodbye!")


if __name__ == "__main__":
    main()
