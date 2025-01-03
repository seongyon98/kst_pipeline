#!/bin/bash
set -e

echo "[Spark Batch Inference] Start..."

# Spark cluster는 docker-compose로 이미 올라와 있다고 가정
# Spark Master 컨테이너에서 spark-submit 실행
docker exec spark-master \
  /app/spark_submit.sh

echo "[Spark Batch Inference] Done!"
