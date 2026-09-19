"""Text preprocessing used by both model training and prediction.

Important:
The exact same preprocessing must be used during training and inference.
Changing tokenization, stopword removal, or lemmatization after a model has
been trained can change the feature space and reduce prediction quality.
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def _ensure_nltk_resources() -> None:
    """Ensure the NLTK corpora required by this project are available.

    NLTK data is not included in the Python package itself. We therefore check
    for the resources explicitly and download them only when they are missing.
    The tokenizer is called with ``preserve_line=True`` below, so the
    punkt/punkt_tab sentence model is not required.
    """
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
    }

    for resource_path, package_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            # ``quiet=True`` prevents noisy download output when the module is
            # imported by FastAPI.
            if not nltk.download(package_name, quiet=True):
                raise RuntimeError(
                    f"Required NLTK resource '{package_name}' is unavailable. "
                    "Install it with: python -m nltk.downloader "
                    f"{package_name}"
                )


_ensure_nltk_resources()

# Initialize the reusable NLP objects once instead of recreating them for
# every prediction request.
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """Normalize an English medicine review for TF-IDF processing.

    Steps:
    1. Convert input to lowercase.
    2. Keep only alphabetic characters and whitespace.
    3. Tokenize the text.
    4. Remove English stopwords.
    5. Lemmatize each remaining token.

    The function intentionally mirrors the preprocessing used when the
    supplied model was trained.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)

    # preserve_line=True avoids NLTK's sentence tokenizer and therefore avoids
    # the common ``punkt_tab`` LookupError in newer NLTK releases.
    tokens = word_tokenize(text, preserve_line=True)

    filtered_tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
        if word not in STOP_WORDS
    ]

    return " ".join(filtered_tokens)
