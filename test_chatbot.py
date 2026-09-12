"""Checks the bot behaves sensibly. Run with:  python test_chatbot.py

Three things are being tested:
  1. Exact and reworded questions find the right answer
  2. Typos still match
  3. Off-topic questions are refused rather than answered with a bad guess

Point 3 is the one that matters most. Without a confidence threshold the bot
will happily explain gradient descent to someone who asked about the weather.
"""

from chatbot import FAQChatbot

# (question asked, expected phrase in the matched FAQ question)
SHOULD_MATCH = [
    ("What is overfitting?", "overfitting"),
    ("whats overfiting", "overfitting"),
    ("Could you explain in detail what overfitting actually means?", "overfitting"),
    ("difference between precision and recall", "precision and recall"),
    ("how do I stop my model memorising the training data", "overfitting"),
    ("what does a learning rate do", "learning rate"),
    ("explain cross validation", "cross-validation"),
    ("why split data into train and test", "training and test"),
    ("what is relu", "ReLU"),
    ("cnn vs rnn", "CNN and an RNN"),
    ("what is tf-idf", "TF-IDF"),
    ("how does backpropagation work", "backpropagation"),
]

# These have no answer in the dataset. The bot must refuse them.
#
# The second group is the harder and more useful half. Plainly off-topic
# questions score 0.000 and would be refused by any threshold at all, so they
# prove very little. These share real vocabulary with the FAQs - "training",
# "model", "network", "data", "learning" - and are what actually pins down how
# high the confidence threshold has to be.
SHOULD_REFUSE = [
    "what is the weather in Mumbai?",
    "who won the cricket match yesterday",
    "can you book me a flight to Delhi",
    "what is your name",
    "tell me a joke",
    # near misses: on-vocabulary, still unanswerable
    "how do I train for a marathon",
    "what is a database index",
    "how much RAM do I need to train a model",
    "what is the learning curve for a new language",
    "how do I network at a tech conference",
    "what salary should I ask for as a data scientist",
]

# Cases this approach cannot get right, kept visible rather than deleted.
#
# "which university has the best machine learning course" contains the exact
# phrase "machine learning", so it matches "What is machine learning?" at 0.62.
# No threshold fixes this: raising it past 0.62 would also refuse most genuine
# questions. TF-IDF matches on shared words, so it can tell what a question is
# *about* but not what it is *asking* - topic and intent look the same to it.
# Getting this right needs sentence embeddings, which understand meaning rather
# than word overlap, and that is a different and much heavier tool than this
# task calls for.
KNOWN_LIMITATIONS = [
    "which university has the best machine learning course",
]


def main() -> int:
    bot = FAQChatbot()
    failures = 0

    print(f"Loaded {len(bot.faqs)} FAQs, threshold {bot.threshold}\n")
    print("--- should match ---")
    for question, expected in SHOULD_MATCH:
        response = bot.ask(question)
        matched = response.matched_question or ""
        ok = response.confident and expected.lower() in matched.lower()
        print(f"{'PASS' if ok else 'FAIL'}  {question!r} -> {matched!r} ({response.score:.3f})")
        if not ok:
            failures += 1

    print("\n--- should refuse ---")
    for question in SHOULD_REFUSE:
        response = bot.ask(question)
        ok = not response.confident
        got = "refused" if ok else f"answered: {response.matched_question!r}"
        print(f"{'PASS' if ok else 'FAIL'}  {question!r} -> {got} ({response.score:.3f})")
        if not ok:
            failures += 1

    print("\n--- known limitations (documented, not counted) ---")
    for question in KNOWN_LIMITATIONS:
        response = bot.ask(question)
        state = "refused" if not response.confident else f"answered: {response.matched_question!r}"
        print(f"KNOWN  {question!r} -> {state} ({response.score:.3f})")

    total = len(SHOULD_MATCH) + len(SHOULD_REFUSE)
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
