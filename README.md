# Medicine Review Sentiment Analysis

A machine-learning application that predicts whether a medicine review is
**Positive** or **Negative** using NLP, TF-IDF feature extraction, and a
Random Forest classifier.

## Architecture

```text
Medicine Review
      |
      v
FastAPI Web UI
      |
      v
Text Preprocessing
(lowercase -> clean -> tokenize -> stopword removal -> lemmatization)
      |
      v
Saved TF-IDF Vectorizer
      |
      v
Saved Random Forest Model
      |
      v
Positive / Negative + Confidence
```

## Project Structure

```text
MEDICINE_REVIEW_PROJECT/
|
├── app.py
├── train.py
├── requirements.txt
├── Dockerfile
├── README.md
|
├── data/
|   └── medicine_reviews.csv
|
├── models/
|   ├── random_forest_model.pkl
|   └── tfidf_vectorizer.pkl
|
├── src/
|   ├── preprocessing.py
|   ├── feature_engineering.py
|   ├── model_training.py
|   └── predict.py
|
├── templates/
|   └── index.html
|
└── static/
    └── style.css
```

## Important: NLTK setup

The preprocessing code requires the NLTK `stopwords` and `wordnet` corpora.

For a local environment, install dependencies and download the data once:

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet
```

The code uses `word_tokenize(..., preserve_line=True)`. This avoids the
`punkt_tab` sentence-tokenizer dependency that commonly causes runtime
`LookupError` exceptions with newer NLTK releases.

The Dockerfile downloads the required NLTK corpora during image creation, so
the running container does not need internet access for NLP resources.

## Train the Model

Run from the project root:

```bash
python train.py
```

This creates:

```text
models/random_forest_model.pkl
models/tfidf_vectorizer.pkl
```

The training script uses a stratified 80/20 train-test split. Stratification is
important here because the supplied dataset is imbalanced toward positive
reviews.

## Run the Web Application

```bash
uvicorn app:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Docker

Build:

```bash
docker build -t medicine-review .
```

Run:

```bash
docker run --rm -p 8000:8000 medicine-review
```

Open:

```text
http://127.0.0.1:8000
```

## Important fixes made

### 1. Fixed the likely prediction-page 500 error

The original preprocessing used:

```python
word_tokenize(text)
```

With newer NLTK versions, this can require the `punkt_tab` resource. The
updated code uses:

```python
word_tokenize(text, preserve_line=True)
```

so sentence-tokenizer data is not required.

The required `stopwords` and `wordnet` corpora are explicitly checked and
downloaded when missing.

### 2. Added `python-multipart` to requirements

FastAPI's `Form(...)` requires `python-multipart`. The original Dockerfile
installed it separately, but a normal local `pip install -r requirements.txt`
did not.

### 3. Fixed confidence display

`predict_proba()` returns a value between 0 and 1. The original UI displayed
that value with a `%` sign, which made the displayed confidence 100 times too
small.

The prediction layer now converts it to 0-100 and rounds it to two decimals.

### 4. Fixed working-directory-dependent paths

Model, template, and static-file paths now use `Path(__file__).resolve()`.
The application therefore works even when Uvicorn is started from a different
current directory.

### 5. Load the model once

The original application loaded the 30 MB Random Forest and TF-IDF vectorizer
for every prediction request. The updated implementation loads them once when
the application module is initialized.

### 6. Added request error handling

Prediction exceptions are logged with a server-side traceback while the UI
shows a controlled error message instead of an unformatted server error.

### 7. Improved training split

The original `train_test_split` did not use stratification. The updated
training code uses:

```python
stratify=df[TARGET_COLUMN]
```

which preserves the class distribution in the train and test sets.

### 8. Reproducibility

The supplied model artifacts were created with scikit-learn 1.9.0. The
requirements therefore pin scikit-learn to 1.9.0 to avoid pickle compatibility
warnings/errors caused by loading the model with another scikit-learn version.

## Dataset observation

The supplied CSV contains 12,112 reviews:

- Positive: 9,191
- Negative: 2,921

That is approximately a 75.9% / 24.1% class split, so accuracy alone should
not be used to judge model quality. Precision, recall, F1-score and the
confusion matrix should also be monitored.

## Model note

The current model is a Random Forest over TF-IDF features. This is a valid
baseline for a text-classification project, but a future experiment could
compare it with Logistic Regression, Linear SVM, or a transformer-based
classifier. Any new model should be evaluated on the same untouched test set
or through a properly designed cross-validation procedure.

## Author

**Kashif**  
Machine Learning | NLP | Data Engineering
