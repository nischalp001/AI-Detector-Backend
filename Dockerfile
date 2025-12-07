# 1. Base Python image
FROM python:3.10-slim

# 2. Install Java (default JRE) and other essentials
RUN apt-get update && \
    apt-get install -y default-jre curl git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy requirements.txt
COPY requirements.txt .

# 5. Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the project
COPY . .

# 7. Download nltk punkt
RUN python -c "import nltk; nltk.download('punkt')"

# 8. Expose FastAPI port
EXPOSE 8000

# 9. Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
