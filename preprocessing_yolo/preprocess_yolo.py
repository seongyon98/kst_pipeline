import boto3
import os
import random

s3 = boto3.client("s3")
BUCKET_NAME = "big9-project-02-training-bucket"
RAW_IMAGES_DIR = "test/images/"
RAW_LABELS_DIR = "test/labels/"
PROCESSED_IMAGES_DIR = "processed_data/images/"
PROCESSED_LABELS_DIR = "processed_data/yolo/"

def transform_to_yolo_format(content):
    """
    bbox 영역만 추출하여 YOLO 형식으로 변환
    """
    bbox_lines = []
    in_bbox_section = False

    for line in content.split("\n"):
        if line.strip() == "[bboxs]":
            in_bbox_section = True
        elif line.strip() == "[question_text]":
            in_bbox_section = False
        elif in_bbox_section:
            bbox_lines.append(line.strip())

    # YOLO 형식 필터링 (class_id, x_center, y_center, width, height)
    yolo_labels = "\n".join([line for line in bbox_lines if len(line.split()) == 5])
    return yolo_labels

def download_and_split_data():
    """
    S3에서 원본 데이터를 다운로드하고 학습/검증 데이터로 분리
    """
    local_image_dir = "/tmp/yolo/images/"
    local_label_dir = "/tmp/yolo/labels/"
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_label_dir, exist_ok=True)

    # 이미지와 라벨 파일 다운로드
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=RAW_IMAGES_DIR)
    image_files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith((".jpg", ".png"))]

    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=RAW_LABELS_DIR)
    label_files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".txt")]

    # 매칭된 이미지와 라벨만 가져오기
    data = list(zip(image_files, label_files))
    random.seed(42)
    random.shuffle(data)

    # 학습/검증 데이터 분리
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    val_data = data[train_size:]

    # S3에 업로드
    upload_split_data(train_data, "train")
    upload_split_data(val_data, "val")

def upload_split_data(data, split_type):
    """
    학습/검증 데이터를 S3로 업로드
    """
    for image_key, label_key in data:
        image_name = os.path.basename(image_key)
        label_name = os.path.basename(label_key)

        # 이미지 업로드
        s3.copy_object(
            Bucket=BUCKET_NAME,
            CopySource={"Bucket": BUCKET_NAME, "Key": image_key},
            Key=f"{PROCESSED_IMAGES_DIR}{split_type}/{image_name}"
        )

        # 라벨 파일 YOLO 형식으로 변환 후 업로드
        label_obj = s3.get_object(Bucket=BUCKET_NAME, Key=label_key)
        content = label_obj["Body"].read().decode("utf-8")
        yolo_result = transform_to_yolo_format(content)

        if yolo_result:
            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=f"{PROCESSED_LABELS_DIR}{split_type}/{label_name}",
                Body=yolo_result.encode("utf-8")
            )
        print(f"Processed {split_type} data saved to S3: {image_name}, {label_name}")

if __name__ == "__main__":
    download_and_split_data()
    print("YOLO 전처리가 완료되었습니다.")
