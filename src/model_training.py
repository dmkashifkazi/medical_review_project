"""Model training and evaluation helpers."""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_model(X_train, y_train):
    """Train a balanced Random Forest classifier.

    ``class_weight='balanced'`` compensates for the dataset's positive/negative
    class imbalance by assigning larger weights to the minority class.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Print standard classification metrics for the held-out test set."""
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Accuracy Score:")
    print(accuracy_score(y_test, y_pred))


def save_model(model, path):
    """Serialize the trained model to disk."""
    joblib.dump(model, path)
    return "Model saved successfully"


def load_model(path):
    """Load a serialized model from disk."""
    return joblib.load(path)
