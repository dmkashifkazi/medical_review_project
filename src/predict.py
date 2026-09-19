"""Inference utilities for the medicine review sentiment model."""

from pathlib import Path

import joblib

from src.preprocessing import preprocess_text


# Build absolute paths from the project root. Relative paths such as
# ``models/...`` can fail when Uvicorn is started from another directory.
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


def _load_artifacts():
    """Load the trained model and TF-IDF vectorizer once.

    Loading the ~30 MB Random Forest on every HTTP request is unnecessary and
    adds latency. Keeping the artifacts in memory also avoids repeated disk I/O.
    """
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not VECTORIZER_PATH.is_file():
        raise FileNotFoundError(f"Vectorizer file not found: {VECTORIZER_PATH}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


MODEL, VECTORIZER = _load_artifacts()


def predict_review(review: str) -> tuple[str, float]:
    """Predict sentiment for one review.

    Returns:
        tuple[str, float]:
            Human-readable prediction label and confidence as a percentage
            between 0 and 100.
    """
    cleaned_review = preprocess_text(review)
    review_vector = VECTORIZER.transform([cleaned_review])

    # The classifier returns the predicted encoded class (0 = negative,
    # 1 = positive for this project).
    prediction = MODEL.predict(review_vector)[0]

    # ``predict_proba`` returns a value between 0 and 1. Convert it to a
    # percentage because the web UI labels the value as "%".
    probability = float(MODEL.predict_proba(review_vector).max() * 100)

    if prediction == 1:
        prediction_label = "Positive Review"
    else:
        prediction_label = "Negative Review"

    return prediction_label, round(probability, 2)
