CREATE DATABASE mlflow_db;
CREATE USER mlflow WITH ENCRYPTED PASSWORD 'mlflow_password';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow;
