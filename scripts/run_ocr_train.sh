#!/bin/bash
set -e

echo "[OCR Training] Start..."
docker build -t ocr-train ./ocr

docker run --rm \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  ocr-train
echo "[OCR Training] Done!"
