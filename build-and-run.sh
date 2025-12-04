#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t onnx-ai-detector-api:latest .

echo "Running container..."
docker run -d \
  --name ai-detector-api \
  -p 8000:8000 \
  -v $(pwd)/onnx-ai-detector:/app/onnx-ai-detector \
  onnx-ai-detector-api:latest

echo "Container is running on http://localhost:8000"
echo "API documentation: http://localhost:8000/docs"