import os
import json
import shutil
import boto3
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
from PIL import Image
from typing import List
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv(override=True)

# AWS 설정
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
MODEL_BUCKET = os.getenv("MODEL_BUCKET")  # 모델이 저장된 S3 버킷 이름

# 로컬 설정
CROPPED_IMAGES_DIR = "./output/craft/cropped"
MODEL_DIR = "./ocr_model"  # 로컬 모델 저장 디렉토리

# S3 클라이언트 초기화
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

# 필수 파일 목록
REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
]


def download_from_s3(bucket: str, key: str, local_path: str):
    """
    S3에서 파일을 다운로드하여 로컬에 저장합니다.
    """
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket, key, local_path)
        print(f"[INFO] S3 파일 다운로드 성공: {key} -> {local_path}")
    except Exception as e:
        print(f"[ERROR] S3 파일 다운로드 실패: {key}, {e}")
        raise


def download_all_model_files(
    bucket: str, s3_prefix: str, local_dir: str, required_files: List[str]
):
    """
    S3에서 모델 파일 전체를 다운로드하고, 필수 파일이 모두 존재하는지 확인합니다.
    """
    print("[INFO] S3에서 모델 파일 다운로드 시작...")
    # S3에서 모델 파일 목록 가져오기
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
    except Exception as e:
        print(f"[ERROR] S3에서 파일 목록 가져오기 실패: {e}")
        return False

    if "Contents" not in response:
        print(f"[ERROR] S3 경로에 파일이 존재하지 않습니다: s3://{bucket}/{s3_prefix}")
        return False

    # 다운로드한 파일 목록 저장
    downloaded_files = set()

    for obj in response["Contents"]:
        s3_key = obj["Key"]
        # 디렉토리는 무시
        if s3_key.endswith("/"):
            continue
        filename = os.path.basename(s3_key)
        local_path = os.path.join(local_dir, filename)
        try:
            download_from_s3(bucket, s3_key, local_path)
            downloaded_files.add(filename)
        except Exception as e:
            print(f"[ERROR] {s3_key} 다운로드 중 오류 발생: {e}")
            continue

    # 필수 파일 존재 여부 확인
    missing_files = [f for f in required_files if f not in downloaded_files]
    if missing_files:
        print(f"[WARN] 누락된 필수 파일: {missing_files}")
        return False
    print("[INFO] 모든 필수 모델 파일이 성공적으로 다운로드되었습니다.")
    return True


def ensure_preprocessor_config_if_missing(model_dir: str):
    """
    모델 디렉토리에 'preprocessor_config.json'이 없을 경우 기본 설정을 생성합니다.
    """
    preproc_json_path = os.path.join(model_dir, "preprocessor_config.json")

    if os.path.exists(preproc_json_path):
        return  # 이미 파일이 존재하면 건너뜀

    print(
        "[WARN] 'preprocessor_config.json' 파일이 없습니다. 기본 설정을 생성합니다..."
    )

    default_config = {
        "model_type": "deit",  # 인코더 구조
        "image_processor_type": "DeiTImageProcessor",
        "do_resize": True,
        "size": 384,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }

    with open(preproc_json_path, "w", encoding="utf-8") as f:
        json.dump(default_config, f, ensure_ascii=False, indent=2)
    print(
        f"[INFO] 기본 'preprocessor_config.json' 파일을 생성했습니다: {preproc_json_path}"
    )


def load_ocr_model_from_s3(
    bucket: str, s3_prefix: str, model_dir: str, required_files: List[str]
):
    """
    S3에서 모델 파일을 다운로드하고, 필수 파일이 모두 존재하면 로컬에서 모델을 로드합니다.
    누락된 파일이 있을 경우 기본 trocr 모델을 로드합니다.
    """
    # 모델 디렉토리 생성
    os.makedirs(model_dir, exist_ok=True)

    # 모델 파일 다운로드
    success = download_all_model_files(bucket, s3_prefix, model_dir, required_files)

    if success:
        # preprocessor_config.json 확인 및 생성
        ensure_preprocessor_config_if_missing(model_dir)

        # 모델 로드 시도
        try:
            model = VisionEncoderDecoderModel.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            image_processor = AutoImageProcessor.from_pretrained(model_dir)
            print("[INFO] S3에서 다운로드한 모델을 로드했습니다.")
            return model, tokenizer, image_processor
        except Exception as e:
            print(f"[ERROR] S3에서 다운로드한 모델 로드 실패: {e}")
            print("[INFO] 기본 trocr 모델을 로드합니다...")
    else:
        print("[INFO] 필수 파일이 누락되어 기본 trocr 모델을 로드합니다...")

    # 필수 파일이 누락된 경우 기본 trocr 모델 로드
    try:
        default_model_name = "team-lucid/trocr-small-korean"
        model = VisionEncoderDecoderModel.from_pretrained(default_model_name)
        tokenizer = AutoTokenizer.from_pretrained(default_model_name)
        image_processor = AutoImageProcessor.from_pretrained(default_model_name)
        print(f"[INFO] 기본 trocr 모델 '{default_model_name}'을 로드했습니다.")
        return model, tokenizer, image_processor
    except Exception as e:
        print(f"[ERROR] 기본 trocr 모델 로드 실패: {e}")
        return None, None, None


def perform_ocr_on_cropped_images(
    image_paths: List[str], model, tokenizer, image_processor, image_size=384
):
    """
    이미지를 OCR 모델을 사용하여 텍스트로 변환하는 함수
    """
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
            decoded_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()

            # 크롭된 이미지 각각에서 추출된 텍스트를 리스트에 추가
            all_texts.append(decoded_text)
            print(f"[INFO] Processed {image_path}: {decoded_text}")

        except Exception as e:
            print(f"[ERROR] Failed to process {image_path}: {e}")
            all_texts.append("")  # 오류 발생 시 빈 텍스트 추가

    return all_texts


import json
import os


def extract_text_from_folders(
    ocr_model, ocr_tokenizer, ocr_processor, cropped_image_base_dir
):
    """
    주어진 폴더에서 OCR을 수행하고 결과를 반환합니다.
    """
    try:
        # 폴더 내의 이미지 파일들을 처리
        image_paths = [
            os.path.join(cropped_image_base_dir, fname)
            for fname in sorted(os.listdir(cropped_image_base_dir))
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_paths:
            print(f"[WARN] No images found in {cropped_image_base_dir}. Skipping...")
            return []

        # OCR 처리
        folder_texts = perform_ocr_on_cropped_images(
            image_paths=image_paths,
            model=ocr_model,
            tokenizer=ocr_tokenizer,
            image_processor=ocr_processor,
            image_size=384,
        )

        # 결과가 리스트 형태일 경우
        if isinstance(folder_texts, list):
            return {"ocr_results": folder_texts}

        # 결과가 단일 문자열일 경우
        elif isinstance(folder_texts, str):
            return {"ocr_results": [{"text": folder_texts}]}

        else:
            raise ValueError("[ERROR] OCR 결과 형식이 예상과 다릅니다.")

    except Exception as e:
        print(f"[ERROR] Failed to process images: {e}")
        return {"ocr_results": []}
