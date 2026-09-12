"""Text preprocessing for the FAQ chatbot.

Stage 1 of the pipeline. Raw human text is messy: "Overfitting", "overfitting?"
and "OVERFITTING" are three different strings to a computer but one idea to us.
This module normalises them so they collapse into the same token.

The steps, in order: lowercase -> strip punctuation -> tokenize -> drop
stopwords -> lemmatize.
"""

import re

import nltk
from nltk.corpus import stopwords, words as nltk_words
from nltk.stem import WordNetLemmatizer

# Words that appear in nearly every question ("what", "is", "the") carry almost
# no signal about which question was asked, so they get dropped. These few are
# kept even though NLTK lists them as stopwords: in an ML FAQ they are load
# bearing. Without this, "supervised vs unsupervised" loses its "not"-style
# contrast words and "how do you prevent overfitting" loses "how".
KEEP_WORDS = {"not", "no", "how", "why", "between", "against", "before", "after", "same", "own"}

# Fired on the first call to preprocess(); NLTK data downloads are slow enough
# that doing it lazily keeps `import preprocess` instant.
_READY = False


def _ensure_nltk_data() -> None:
    """Download the NLTK corpora we need, once, if they are not already there."""
    global _READY
    if _READY:
        return
    for path, package in [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/words", "words"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)
    _READY = True


class TextPreprocessor:
    """Turns a raw sentence into a cleaned list of meaningful root words."""

    def __init__(self) -> None:
        _ensure_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words("english")) - KEEP_WORDS
        # An English dictionary, used by the chatbot's spell corrector to tell a
        # typo apart from a real word it simply has no FAQ about.
        self.english_words = {w.lower() for w in nltk_words.words()}

    def tokenize(self, text: str) -> list[str]:
        """Lowercase, then pull out runs of letters and digits as tokens.

        A regex is used rather than nltk.word_tokenize because it needs no
        extra downloaded model and, for this job, punctuation is being thrown
        away anyway. "don't" becomes ["don", "t"], and the stray "t" is dropped
        by the length filter below.
        """
        return re.findall(r"[a-z0-9]+", text.lower())

    def process(self, text: str) -> list[str]:
        """Run the full pipeline and return the surviving root words."""
        tokens = self.tokenize(text)
        cleaned = []
        for token in tokens:
            if token in self.stopwords:
                continue
            if len(token) < 2:  # stray letters left over from splitting "don't", "it's"
                continue
            # Lemmatize as a noun first, then as a verb. Doing both catches
            # "models" -> "model" and "learning" -> "learn"; the noun-only
            # default would leave every verb form untouched.
            root = self.lemmatizer.lemmatize(token, pos="n")
            root = self.lemmatizer.lemmatize(root, pos="v")
            cleaned.append(root)
        return cleaned

    def process_to_string(self, text: str) -> str:
        """Same as process(), joined back into a string.

        TfidfVectorizer wants strings, not token lists, so this is the form the
        vectorizer actually consumes.
        """
        return " ".join(self.process(text))


if __name__ == "__main__":
    # Run `python preprocess.py` to watch the cleaning happen. Printing the
    # intermediate values is the fastest way to see why a match succeeded
    # or failed.
    pre = TextPreprocessor()
    samples = [
        "What is overfitting?",
        "whats overfiting",
        "Could you please explain to me what OVERFITTING actually means?",
        "How are models evaluated using cross-validation?",
    ]
    for sample in samples:
        print(f"{sample!r}\n  -> {pre.process(sample)}\n")
