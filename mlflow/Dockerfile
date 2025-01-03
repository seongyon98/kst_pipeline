FROM python:3.9-slim

RUN pip install --no-cache-dir mlflow boto3 psycopg2-binary

WORKDIR /mlflow
EXPOSE 5000

# CMD에 환경변수를 참조하도록 수정
CMD ["mlflow", "server", "--backend-store-uri", "${BACKEND_STORE_URI}","--default-artifact-root", "${ARTIFACT_STORE}","--host", "0.0.0.0","--port", "5000"]
