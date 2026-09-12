"""The FAQ chatbot itself: matching logic, no interface.

Stages 2 to 4 of the pipeline live here. The bot does not generate answers and
does not understand English. It holds a fixed list of question-answer pairs and
its only job is to work out which stored question is closest to what the user
typed, then hand back that question's answer.

Keeping this file free of any UI code means the same class serves both the web
app and the terminal version, and can be tested on its own. The interface was
swapped from Streamlit to Flask without touching a line of this file.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import TextPreprocessor

# Below this cosine score the bot says it does not know rather than serving its
# best bad guess.
#
# Both numbers below come from a grid search over question_weight x threshold,
# scored against the cases in test_chatbot.py. 3 and 0.25 was the best cell at
# 23/24. Lower thresholds start letting near-miss questions through ("how do I
# train for a marathon" matches training-set FAQs at 0.20); higher ones start
# refusing wordy but answerable questions.
CONFIDENCE_THRESHOLD = 0.25

# How many times a FAQ's question is repeated relative to its answer when
# building the text that FAQ is indexed against. See _build_document().
QUESTION_WEIGHT = 3

# Longer input than this is truncated before matching.
#
# Spell correction compares each unknown word against the whole vocabulary, so
# cost grows with the number of unrecognised words: measured at ~0.3 ms each,
# meaning a pasted 2000-word block took 546 ms while a real question takes under
# 1 ms. No genuine FAQ question is anywhere near 500 characters, so cutting the
# input costs nothing and keeps the response time flat.
MAX_QUESTION_CHARS = 500

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "faqs.json"

FALLBACK_MESSAGE = (
    "I don't know that one yet. I only cover AI and machine learning concepts. "
    "Try rephrasing, or ask about something like overfitting, gradient descent "
    "or precision and recall."
)

# Greetings and pleasantries, handled before the FAQ match runs.
#
# "hi" is the first thing most people type, and matching it against the FAQ list
# produces a cosine score near zero and the "I don't know" fallback - technically
# correct and a terrible first impression. These are intents, not FAQs, so they
# live here rather than being faked as entries in faqs.json where they would
# pollute the vocabulary and the IDF weights.
#
# Matching is on the whole cleaned input, not on individual words. A user who
# types just "hi" is greeting; a user who types "what is a hidden layer" is not,
# even though it starts with the same two letters. Requiring the entire input to
# match makes a false positive impossible.
SMALL_TALK = {
    "greeting": (
        {
            "hi", "hii", "hiii", "hello", "helo", "hey", "heya", "hiya", "yo",
            "greetings", "hi there", "hello there", "hey there", "good morning",
            "good afternoon", "good evening", "gm", "sup", "whats up",
        },
        "Hello. Ask me about any AI or machine learning concept: overfitting, "
        "gradient descent, precision and recall, and so on.",
    ),
    "thanks": (
        {
            "thanks", "thank you", "thankyou", "thanks a lot", "thanks so much",
            "thank you very much", "ty", "tysm", "cheers", "appreciate it", "nice",
            "cool", "great", "awesome", "perfect", "helpful", "that helps",
        },
        "Glad that helped. Ask me another one whenever you like.",
    ),
    "goodbye": (
        {"bye", "goodbye", "bye bye", "see you", "see ya", "cya", "later", "good night"},
        "Bye. Good luck with the revision.",
    ),
    "identity": (
        {
            "who are you", "what are you", "what can you do", "help", "what do you know",
            "how do you work", "what is this", "what can i ask", "what can i ask you",
        },
        "I'm an FAQ bot for AI and machine learning concepts. I don't generate "
        "answers. I hold a fixed list of question-answer pairs and use TF-IDF and "
        "cosine similarity to find the stored question closest to yours. If nothing "
        "is close enough, I say so instead of guessing.",
    ),
}


@dataclass
class Match:
    """One scored candidate answer."""

    question: str
    answer: str
    score: float


@dataclass
class Response:
    """What the bot hands back for a single user question.

    `confident` is the interesting field. When it is False the caller should
    show `text` (the fallback) rather than pretending to have an answer, and
    may offer `suggestions` as near misses.
    """

    text: str
    confident: bool
    score: float
    matched_question: str | None = None
    suggestions: list[Match] | None = None
    # "answer" - matched a FAQ; "small_talk" - a greeting or pleasantry, handled
    # without the matcher; "fallback" - nothing scored above the threshold.
    kind: str = "answer"


class FAQChatbot:
    def __init__(
        self,
        data_path: Path | str = DEFAULT_DATA_PATH,
        threshold: float = CONFIDENCE_THRESHOLD,
        question_weight: int = QUESTION_WEIGHT,
    ) -> None:
        self.threshold = threshold
        self.question_weight = question_weight
        self.preprocessor = TextPreprocessor()

        with open(data_path, encoding="utf-8") as f:
            self.faqs = json.load(f)

        # Checked up front with a readable message. The FAQ file is the one part
        # of this project meant to be edited by someone who is not reading the
        # code, so a malformed entry should say which entry and what is wrong,
        # not surface later as a bare KeyError from a list comprehension.
        if not isinstance(self.faqs, list) or not self.faqs:
            raise ValueError(f"{data_path} must contain a non-empty JSON list of FAQs")
        for i, faq in enumerate(self.faqs):
            if not isinstance(faq, dict):
                raise ValueError(f"FAQ {i} in {data_path} is not an object")
            missing = {"question", "answer"} - faq.keys()
            if missing:
                raise ValueError(f"FAQ {i} in {data_path} is missing: {', '.join(sorted(missing))}")
            if not str(faq["question"]).strip() or not str(faq["answer"]).strip():
                raise ValueError(f"FAQ {i} in {data_path} has an empty question or answer")

        self.questions = [faq["question"] for faq in self.faqs]
        self.answers = [faq["answer"] for faq in self.faqs]

        # Each FAQ is indexed as its question plus its answer, cleaned down to
        # root words. Including the answer lets a question phrased in the
        # answer's own words still find it - "how do I stop my model memorising
        # the training data" shares no useful word with the question "What is
        # overfitting?" but plenty with its answer.
        processed = [self._build_document(q, a) for q, a in zip(self.questions, self.answers)]

        # Every word the bot knows. Used to spell-correct user input below.
        self.vocabulary = {word for text in processed for word in text.split()}

        # ngram_range=(1, 2) makes the vectorizer track word pairs as well as
        # single words, so "cross validation" and "learning rate" are matched
        # as phrases rather than as two independent words that happen to co-occur.
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.faq_vectors = self.vectorizer.fit_transform(processed)

    def _build_document(self, question: str, answer: str) -> str:
        """Cleaned text that this FAQ is matched against.

        The question is repeated `question_weight` times before the answer is
        appended. Term frequency counts repeats, so this makes question words
        carry more weight than answer words without needing a custom vectorizer
        - the answer broadens what can be found, the repetition keeps the
        question in charge of the match.
        """
        cleaned_question = self.preprocessor.process_to_string(question)
        cleaned_answer = self.preprocessor.process_to_string(answer)
        return " ".join([cleaned_question] * self.question_weight + [cleaned_answer])

    def _correct_spelling(self, tokens: list[str]) -> list[str]:
        """Nudge misspelled words onto the nearest word the bot actually knows.

        TF-IDF matches on exact string equality, so "overfiting" scores zero
        against "overfitting" - a one-character typo makes a question
        unanswerable.

        The dictionary check on the second line is the important part. Without
        it, correcting purely by string distance rewrites real words into FAQ
        vocabulary: "who won the cricket match" had its "match" turned into
        "batch", which then matched the mini-batch gradient descent FAQ at 0.48
        and defeated the confidence threshold. A word that is in the English
        dictionary is left alone - it is a topic the bot has no FAQ about, not
        a typo.
        """
        corrected = []
        for token in tokens:
            if token in self.vocabulary or token in self.preprocessor.english_words:
                corrected.append(token)
                continue
            close = difflib.get_close_matches(token, self.vocabulary, n=1, cutoff=0.8)
            corrected.append(close[0] if close else token)
        return corrected

    def rank(self, user_question: str, top_n: int = 3) -> list[Match]:
        """Score the question against every stored FAQ, best first."""
        tokens = self._correct_spelling(self.preprocessor.process(user_question))
        query_vector = self.vectorizer.transform([" ".join(tokens)])

        # One score per stored question, each between 0 and 1.
        scores = cosine_similarity(query_vector, self.faq_vectors)[0]

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            Match(self.questions[i], self.answers[i], float(scores[i]))
            for i in ranked[:top_n]
        ]

    @staticmethod
    def _small_talk_reply(user_question: str) -> str | None:
        """Return a canned reply if the whole input is a greeting or pleasantry."""
        # Lowercase, drop anything that is not a letter or space, collapse runs
        # of whitespace: "Hi there!!" and "  hi   there " both become "hi there".
        normalised = re.sub(r"[^a-z\s]", "", user_question.lower())
        normalised = " ".join(normalised.split())
        if not normalised:
            return None
        for phrases, reply in SMALL_TALK.values():
            if normalised in phrases:
                return reply
        return None

    def ask(self, user_question: str) -> Response:
        """Answer a question, or admit that it cannot."""
        if not user_question.strip():
            return Response(
                text="Ask me something about AI or ML.", confident=False, score=0.0,
                kind="fallback",
            )

        user_question = user_question[:MAX_QUESTION_CHARS]

        small_talk = self._small_talk_reply(user_question)
        if small_talk:
            return Response(text=small_talk, confident=True, score=0.0, kind="small_talk")

        matches = self.rank(user_question)
        best = matches[0]

        if best.score < self.threshold:
            # Only offer near misses that scored something; suggesting a
            # question that scored 0.01 is noise, not a hint.
            suggestions = [m for m in matches if m.score > 0.1]
            return Response(
                text=FALLBACK_MESSAGE,
                confident=False,
                score=best.score,
                suggestions=suggestions or None,
                kind="fallback",
            )

        return Response(
            text=best.answer,
            confident=True,
            score=best.score,
            matched_question=best.question,
        )
