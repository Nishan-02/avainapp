FROM python:3.11-slim

# Install system dependencies required for librosa and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port (FastAPI default, Railway injects PORT environment variable)
EXPOSE 8080

# Run the application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
