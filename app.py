"""FastAPI web application for medicine review sentiment prediction.

The application exposes a small HTML UI and a POST endpoint that accepts a
medicine review and returns the sentiment predicted by the trained model.
"""

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.predict import predict_review


# Resolve paths from this file instead of relying on the process working
# directory. This makes the application work consistently from local shells,
# IDEs, Docker, and process managers.
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Medicine Review Sentiment Analysis",
    description="Predicts whether a medicine review is positive or negative.",
)

# Serve CSS/JS/images from the static directory.
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

# Configure Jinja2 to render the web UI.
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/")
async def read_root(request: Request):
    """Render the prediction form."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/predict")
async def predict(request: Request, review: str = Form(...)):
    """Preprocess the submitted review and return its predicted sentiment.

    The prediction function returns a human-readable label and a probability
    already converted to a percentage (0-100).
    """
    # Strip leading/trailing whitespace before sending the text to the model.
    review = review.strip()

    # Give the user a clear validation message instead of allowing an empty
    # review to reach the ML pipeline.
    if not review:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "review": review,
                "error": "Please enter a medicine review before clicking Predict.",
            },
            status_code=400,
        )

    try:
        prediction, probability = predict_review(review)
    except Exception:
        # Log the full traceback on the server while showing a safe, useful
        # message in the browser. Avoid exposing filesystem paths/model details.
        import logging

        logging.exception("Prediction failed")
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "review": review,
                "error": (
                    "Prediction could not be completed. "
                    "Check the application console/logs for the detailed error."
                ),
            },
            status_code=500,
        )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "review": review,
            "prediction": prediction,
            "probability": probability,
        },
    )
