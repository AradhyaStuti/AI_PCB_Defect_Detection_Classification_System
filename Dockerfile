# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System libs needed by scikit-image / Pillow, plus curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Deps first so they stay cached across source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs

# 8000 = REST API, 8501 = Streamlit.
EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
