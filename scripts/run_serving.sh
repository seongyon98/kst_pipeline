#!/bin/bash
set -e

echo "[Serving] Start..."
docker-compose up -d serving
echo "[Serving] Done! Access at http://localhost:8000/"
