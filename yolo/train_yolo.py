from ultralytics import YOLO
import boto3
import os
import yaml
import mlflow
import mlflow.pytorch

# S3 client
s3 = boto3.client("s3")

# Constants
#BUCKET_NAME = "big9-project-02-training-bucket"
PROCESSED_IMAGES_DIR = "processed_data/images/"
PROCESSED_LABELS_DIR = "processed_data/yolo/"
LOCAL_DATASET_DIR = "/tmp/yolo_dataset/"
DATA_YAML_PATH = os.path.join(LOCAL_DATASET_DIR, "data.yaml")
#MLFLOW_TRACKING_URI = "http://3.38.40.255:5000"  # MLflow 서버 주소
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "big9-project-02-training-bucket")

def download_s3_directory(bucket_name, prefix, local_dir):
    """
    S3 디렉토리 내 파일 다운로드
    """
    os.makedirs(local_dir, exist_ok=True)
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if "Contents" not in response:
        print(f"No files found in {prefix}")
        return

    for obj in response["Contents"]:
        file_key = obj["Key"]
        if file_key.endswith('/'):
            continue  # 디렉토리인 경우 건너뜀
        file_name = os.path.basename(file_key)
        local_path = os.path.join(local_dir, file_name)
        s3.download_file(bucket_name, file_key, local_path)
        print(f"Downloaded {file_key} to {local_path}")

def prepare_dataset():
    """
    S3에서 데이터를 다운로드하고 data.yaml 생성
    """
    print("Preparing YOLO dataset...")

    # 학습/검증 데이터 경로
    train_images_dir = os.path.join(LOCAL_DATASET_DIR, "images/train")
    val_images_dir = os.path.join(LOCAL_DATASET_DIR, "images/val")
    train_labels_dir = os.path.join(LOCAL_DATASET_DIR, "labels/train")
    val_labels_dir = os.path.join(LOCAL_DATASET_DIR, "labels/val")

    # 이미지와 라벨 다운로드
    download_s3_directory(BUCKET_NAME, f"{PROCESSED_IMAGES_DIR}train/", train_images_dir)
    download_s3_directory(BUCKET_NAME, f"{PROCESSED_IMAGES_DIR}val/", val_images_dir)
    download_s3_directory(BUCKET_NAME, f"{PROCESSED_LABELS_DIR}train/", train_labels_dir)
    download_s3_directory(BUCKET_NAME, f"{PROCESSED_LABELS_DIR}val/", val_labels_dir)

    # data.yaml 생성
    data_config = {
        "path": LOCAL_DATASET_DIR,
        "train": "images/train",
        "val": "images/val",
        "names": ["text", "non-text"]
    }
    with open(DATA_YAML_PATH, "w") as file:
        yaml.dump(data_config, file)
    print(f"data.yaml created at {DATA_YAML_PATH}")

def train_yolo():
    """
    YOLO 모델 학습
    """
    print("Starting YOLO training...")
    

    # MLflow 설정
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("YOLO Training Pipeline")

    model = YOLO("yolov8n.pt")  # Pretrained YOLOv8 모델 로드

    with mlflow.start_run():
        # YOLO 학습
        results = model.train(
            data=DATA_YAML_PATH,
            epochs=1,  # 학습 에포크
            imgsz=640,
            batch=8,
            device="cpu",  # GPU 사용 가능 시 'cuda'
            name="yolo_training2"  # 결과 저장 디렉토리 이름
        )
        print("YOLO training completed.")

        # YOLO 결과 디렉토리 확인
        trained_model_dir = os.path.join(results.save_dir, "weights")
        print(f"Trained model directory: {trained_model_dir}")
        if not os.path.exists(trained_model_dir):
            raise FileNotFoundError(f"Trained model directory not found at {trained_model_dir}")

        # YOLO 결과 경로에서 모델 파일 경로 찾기
        trained_model_path = os.path.join(trained_model_dir, "best.pt")
        if not os.path.exists(trained_model_path):
            print(f"File 'best.pt' not found. Checking for 'last.pt' instead.")
            trained_model_path = os.path.join(trained_model_dir, "last.pt")

        if not os.path.exists(trained_model_path):
            raise FileNotFoundError(f"Trained model not found at {trained_model_path}")

        # 학습된 모델 S3 업로드
        # s3.upload_file(trained_model_path, BUCKET_NAME, "models/yolov8n_best.pt")
        # print(f"Trained model uploaded to s3://{BUCKET_NAME}/models/yolov8n_best.pt")
        
        # MLflow에 메트릭 및 아티팩트 기록
        mlflow.log_metric("precision", results.box.mp)  # Mean Precision
        mlflow.log_metric("recall", results.box.mr)    # Mean Recall
        mlflow.log_metric("mAP50", results.box.map50)  # Mean AP at IoU 0.5
        mlflow.log_metric("mAP50-95", results.box.map) # Mean AP from 0.5 to 0.95

        mlflow.log_param("epochs", 1)
        mlflow.log_param("batch_size", 8)

        # 학습된 모델 파일 로깅
        mlflow.log_artifact(trained_model_path, artifact_path="model_weights")

if __name__ == "__main__":
    prepare_dataset()
    train_yolo()
