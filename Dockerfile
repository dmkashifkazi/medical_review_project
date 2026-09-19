FROM python:3.13-slim

# Keep the container self-contained and reproducible.
WORKDIR /app

# Install Python dependencies first so Docker can cache this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the NLTK corpora required by preprocessing.py at build time.
# This prevents the running API from depending on outbound internet access.
RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords wordnet

ENV NLTK_DATA=/usr/local/share/nltk_data
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy application source, templates, static assets, data, and model artifacts.
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]