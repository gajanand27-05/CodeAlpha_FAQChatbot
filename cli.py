"""Terminal version of the FAQ chatbot.

Run with:  python cli.py
Useful for testing the matching logic without the web layer in the way, and for
recording a quick demo. Type 'quit' to exit, or 'debug' to toggle showing the
match score and which stored question was hit.
"""

from chatbot import FAQChatbot


def main() -> None:
    print("Loading FAQs...")
    bot = FAQChatbot()
    print(f"Ready - {len(bot.faqs)} questions loaded.")
    print("Ask me about AI/ML concepts. Type 'quit' to exit, 'debug' to see scores.\n")

    debug = False
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Bye.")
            return
        if user_input.lower() == "debug":
            debug = not debug
            print(f"[debug {'on' if debug else 'off'}]\n")
            continue

        response = bot.ask(user_input)
        print(f"\nBot: {response.text}")

        if debug:
            print(f"     [score {response.score:.3f}, threshold {bot.threshold}]")
            if response.matched_question:
                print(f"     [matched: {response.matched_question}]")

        if response.suggestions:
            print("\n     Did you mean one of these?")
            for match in response.suggestions:
                print(f"       - {match.question}")
        print()


if __name__ == "__main__":
    main()
