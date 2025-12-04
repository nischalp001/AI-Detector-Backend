FROM python:3.10-slim-bookworm

WORKDIR /app

# Install system dependencies (minimal, all at once)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Verify Java installation (JRE only, javac not needed)
RUN java -version

# Copy requirements early (better layer caching)
COPY requirements.txt .

# Install Python dependencies (single RUN to reduce layers)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data during build (not at runtime)
RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/local/share/nltk_data')"

# Copy application code
COPY app.py .
COPY onnx-ai-detector/ ./onnx-ai-detector/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]