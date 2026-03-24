# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System libraries required by scikit-image / Pillow and the health-check curl
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first — maximises Docker layer cache reuse
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Persistent log directory
RUN mkdir -p logs

# REST API port
EXPOSE 8000
# Streamlit UI port
EXPOSE 8501

# Default entry-point: REST API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
