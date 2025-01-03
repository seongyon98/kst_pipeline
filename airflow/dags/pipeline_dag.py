from airflow import DAG
from airflow.operators.docker_operator import DockerOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    "yolo_pipeline",
    default_args=default_args,
    description="YOLO Training Pipeline",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    preprocess_task = DockerOperator(
        task_id="preprocess_yolo",
        image="yolo_preprocessing_image",
        auto_remove=True,
        command="python preprocess_yolo.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge"
    )

    train_task = DockerOperator(
        task_id="train_yolo",
        image="yolo_training_image",
        auto_remove=True,
        command="python train_yolo.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge"
    )

    preprocess_task >> train_task
