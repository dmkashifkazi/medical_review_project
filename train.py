"""Train the medicine-review sentiment classifier.

Run from the project root:

    python train.py

The script creates a TF-IDF vectorizer and Random Forest model, evaluates them
on a held-out test split, and saves both artifacts under ``models/``.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_engineering import create_tfidf, save_vectorizer, transform_text
from src.model_training import evaluate_model, save_model, train_model
from src.preprocessing import preprocess_text


# Resolve all project paths relative to this file so the script does not depend
# on the directory from which Python happens to be launched.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "medicine_reviews.csv"
MODEL_FOLDER = BASE_DIR / "models"

TEXT_COLUMN = "review"
TARGET_COLUMN = "label"


def main():
    """Load data, train the pipeline, evaluate it, and save model artifacts."""
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Keep only rows that contain both the review text and its sentiment label.
    df = df.dropna(subset=[TEXT_COLUMN, TARGET_COLUMN]).copy()

    # Apply the same text normalization that will be used during inference.
    print("Preprocessing reviews...")
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(preprocess_text)

    # Convert human-readable labels to the numeric classes expected by the
    # Random Forest: positive=1 and negative=0.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(
        {"positive": 1, "negative": 0}
    )

    # Remove any unexpected labels rather than silently training with NaN
    # targets. Stratification preserves the class ratio in train and test sets.
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COLUMN],
        df[TARGET_COLUMN],
        test_size=0.20,
        random_state=42,
        stratify=df[TARGET_COLUMN],
    )

    print("Creating TF-IDF features...")
    vectorizer, X_train_tfidf = create_tfidf(X_train)
    X_test_tfidf = transform_text(vectorizer, X_test)

    print("Training Random Forest...")
    model = train_model(X_train_tfidf, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test_tfidf, y_test)

    print("Saving model artifacts...")
    save_model(model, MODEL_FOLDER / "random_forest_model.pkl")
    save_vectorizer(vectorizer, MODEL_FOLDER / "tfidf_vectorizer.pkl")

    print("Model and vectorizer saved successfully.")


if __name__ == "__main__":
    main()
