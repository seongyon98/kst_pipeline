import os
import boto3
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
from typing import List
from dotenv import load_dotenv
from yolo_api.src.yolo_model import CROPPED_IMAGES_DIR  # 크롭된 이미지 폴더 경로

# 환경 변수 로드
load_dotenv(dotenv_path='/pipeline/.env', override=True)

# AWS 자격 증명
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
MODEL_BUCKET = os.getenv("MODEL_BUCKET")
OCR_S3_KEY = "ocr/final_model/final_model_1/" 

# S3 클라이언트 초기화
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

# ---------------------------------------------------------------------
# 1. OCR 모델 S3에서 바로 로드
# ---------------------------------------------------------------------
def load_ocr_model_from_s3(bucket: str, s3_prefix: str):
    """
    S3에서 OCR 모델을 바로 로드
    """
    try:
        s3_path = f"s3://{bucket}/{s3_prefix}"
        print(f"[INFO] Loading OCR model directly from S3: {s3_path}")
        
        model = VisionEncoderDecoderModel.from_pretrained(s3_path)
        tokenizer = AutoTokenizer.from_pretrained(s3_path)
        image_processor = AutoImageProcessor.from_pretrained(s3_path)
        return model, tokenizer, image_processor
    except Exception as e:
        print(f"[ERROR] Failed to load OCR model from S3: {e}")
        return None, None, None

# ---------------------------------------------------------------------
# 2. 크롭된 이미지 리스트를 받아 OCR을 수행하고 결과를 반환
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
            all_texts.append("")  # 오류 발생 시 빈 텍스트 추가

    return all_texts

# ---------------------------------------------------------------------
# 3. 메인 프로세스
# ---------------------------------------------------------------------
def main():
    # 1) S3에서 OCR 모델 로드
    model, tokenizer, image_processor = load_ocr_model_from_s3(MODEL_BUCKET, OCR_S3_KEY)
    if not model or not tokenizer or not image_processor:
        print("[ERROR] OCR 모델 로드에 실패했습니다.")
        return

    # 2) 크롭된 이미지 경로 리스트 생성
    if not os.path.exists(CROPPED_IMAGES_DIR):
        print(f"[ERROR] Cropped 이미지 폴더가 없습니다: {CROPPED_IMAGES_DIR}")
        return

    image_paths = [
        os.path.join(CROPPED_IMAGES_DIR, fname)
        for fname in os.listdir(CROPPED_IMAGES_DIR)
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