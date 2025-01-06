import os
import boto3
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
from typing import List
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv(dotenv_path='/pipeline/.env', override=True)

# AWS 자격 증명
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
MODEL_BUCKET = os.getenv("MODEL_BUCKET")
OCR_S3_KEY = "ocr/final_model/final_model_1/" 

# 로컬 OCR 모델 및 이미지 디렉토리
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
OCR_LOCAL_PATH = os.path.join(MODEL_DIR, "final_model_1")
CROPPED_IMAGES_PATH = os.path.join(BASE_DIR, "cropped_images")  # 크롭된 이미지 폴더 경로 변경

# S3 클라이언트 초기화
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

# ---------------------------------------------------------------------
# 1. OCR 모델 S3 -> 로컬 다운로드
# ---------------------------------------------------------------------
def download_s3_directory(bucket, prefix, local_dir):
    paginator = s3_client.get_paginator("list_objects_v2")
    print(f"[INFO] Downloading OCR model directory from s3://{bucket}/{prefix} ...")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        for obj in contents:
            key = obj["Key"]
            if key.endswith("/"):  # 폴더 Key 는 제외
                continue

            relative_path = key[len(prefix):]  # prefix 이후의 경로만 추출
            local_file_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)  # "model" 폴더가 없으면 만들기

            if os.path.exists(local_file_path):
                print(f"[INFO] Local file already exists: {local_file_path}. Skipping.")
                continue

            print(f"[INFO] Downloading s3://{bucket}/{key} -> {local_file_path}")
            s3_client.download_file(bucket, key, local_file_path)

# OCR 모델 디렉토리 확인 및 다운로드
if os.path.exists(OCR_LOCAL_PATH):
    print(f"[INFO] OCR model directory already exists: {OCR_LOCAL_PATH}. Skipping download.")
else:
    download_s3_directory(MODEL_BUCKET, OCR_S3_KEY, OCR_LOCAL_PATH)
    print(f"[INFO] OCR model directory downloaded to: {OCR_LOCAL_PATH}")

# ---------------------------------------------------------------------
# 2. OCR 모델, 토크나이저, 이미지 프로세서 로드
# ---------------------------------------------------------------------
def load_ocr_model(model_dir):
    print(f"[INFO] Loading OCR model, tokenizer, and processor from: {model_dir}")
    try:
        model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        image_processor = AutoImageProcessor.from_pretrained(model_dir)
        return model, tokenizer, image_processor
    except Exception as e:
        print(f"[ERROR] Failed to load OCR components: {e}")
        return None, None, None

# ---------------------------------------------------------------------
# 3. 크롭된 이미지 리스트를 받아 OCR을 수행하고 결과를 반환
# ---------------------------------------------------------------------
def perform_ocr_on_cropped_images(image_paths: List[str], model, tokenizer, image_processor, image_size=384):
    all_texts = []
    for image_path in image_paths:
        try:
            # 이미지 로드 및 리사이즈
            img = Image.open(image_path).convert("RGB")
            img = img.resize((image_size, image_size))

            # 이미지 전처리
            pixel_values = image_processor(images=img, return_tensors="pt").pixel_values

            # OCR 모델 추론
            output_ids = model.generate(pixel_values, max_length=512)
            decoded_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            # all_texts: 크롭된 이미지 각각에서 추출된 개별 텍스트를 담는 리스트
            all_texts.append(decoded_text)
            print(f"[INFO] Processed {image_path}: {decoded_text}")

        except Exception as e:
            print(f"[ERROR] Failed to process {image_path}: {e}")
            all_texts.append("")  # 오류 발생 시 빈 텍스트 추가                                                                                     nd("")  # 오류 발생 시 빈 텍스트 추가

    return all_texts

# ---------------------------------------------------------------------
# 4. 메인 프로세스
# ---------------------------------------------------------------------
def main():
    # 1) OCR 모델 로드
    model, tokenizer, image_processor = load_ocr_model(OCR_LOCAL_PATH)
    if not model or not tokenizer or not image_processor:
        print("[ERROR] OCR 모델 로드에 실패했습니다.")
        return

    # 2) 크롭된 이미지 경로 리스트 생성
    if not os.path.exists(CROPPED_IMAGES_PATH):
        print(f"[ERROR] Cropped 이미지 폴더가 없습니다: {CROPPED_IMAGES_PATH}")
        return

    image_paths = [
        os.path.join(CROPPED_IMAGES_PATH, fname)
        for fname in os.listdir(CROPPED_IMAGES_PATH)
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_paths:
        print("[ERROR] 크롭된 이미지가 없습니다.")
        return

    # 3) OCR 수행
    all_texts = perform_ocr_on_cropped_images(
        image_paths=image_paths,
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_size=384,
    )

    # 4) all_texts 에 저장된 개별 텍스트를 하나의 문자열로 합침
    final_text = " ".join(all_texts).strip()
    print(f"\n[RESULT] Combined text for LLM: {final_text}")

if __name__ == "__main__":
    main()

# LLM 한테 전달할 최종 텍스트: "final_text" 