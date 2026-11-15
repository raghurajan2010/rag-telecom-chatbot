"""CLI REPL for the telecom RAG chatbot. Type 'quit' to exit."""
import sys

from rag_chain import TelecomRAG, Timer, log_interaction


def main():
    # Avoid crashes on Windows consoles that can't encode emoji/unicode in model output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print("Loading knowledge base and models...")
    bot = TelecomRAG()
    print("NovaCell Telecom Assistant - ask a question, or type 'quit' to exit.\n")
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
        timer = Timer()
        docs = bot.retrieve(question)
        print("Bot: ", end="", flush=True)
        answer, first = "", None
        for token in bot.stream_answer(question, docs):
            first = first if first is not None else timer.elapsed()
            answer += token
            sys.stdout.write(token)
            sys.stdout.flush()
        print("\n\nSources:")
        for d in docs:
            print(f"  - {d.metadata.get('source')}: {d.metadata.get('ref')}")
        print()
        log_interaction(question, answer, docs, timer.elapsed(), first or 0.0, "cli")
    print("Goodbye!")


if __name__ == "__main__":
    main()
