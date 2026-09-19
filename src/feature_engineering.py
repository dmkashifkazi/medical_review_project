"""TF-IDF feature engineering helpers."""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf(X_train):
    """Fit a TF-IDF vectorizer on training text and transform that text.

    The vectorizer must be fitted only on the training split to avoid test-set
    information leaking into the feature-generation step.
    """
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)

    return vectorizer, X_train_tfidf


def transform_text(vectorizer, X_test):
    """Transform new text with an already-fitted TF-IDF vectorizer."""
    return vectorizer.transform(X_test)


def save_vectorizer(vectorizer, path):
    """Serialize a fitted TF-IDF vectorizer to disk."""
    joblib.dump(vectorizer, path)
    return "Vectorizer saved successfully"


def load_vectorizer(path):
    """Load a previously saved TF-IDF vectorizer."""
    return joblib.load(path)
