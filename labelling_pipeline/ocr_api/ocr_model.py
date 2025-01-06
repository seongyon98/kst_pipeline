# [로컬에서 ocr 모델 로드]
import os
import json
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
from typing import List
from yolo_api.src.config import CROPPED_IMAGES_DIR  # 크롭된 이미지 폴더 경로

# 로컬 모델 경로
LOCAL_MODEL_PATH = "C:/Users/user/Desktop/final_project/github/labelling_pipeline/ocr_api/models/final_model_1"

# ---------------------------------------------------------------------
# 1. OCR 모델 로컬에서 로드
# ---------------------------------------------------------------------
def ensure_preprocessor_config_if_missing(model_dir):
    """
    모델 디렉토리에 'preprocessor_config.json'이 없을 경우 기본 설정 생성
    """
    preproc_json_path = os.path.join(model_dir, "preprocessor_config.json")

    if os.path.exists(preproc_json_path):
        return  # 이미 파일이 존재하면 건너뜀

    print("[WARN] Missing 'preprocessor_config.json'. Creating default config...")

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
    print(f"[INFO] Created 'preprocessor_config.json' at {preproc_json_path}")

def load_ocr_model(model_path: str):
    """
    로컬에서 OCR 모델을 로드
    """
    try:
        print(f"[INFO] Loading OCR model from local path: {model_path}")

        if not os.path.exists(model_path):
            print(f"[ERROR] Model path does not exist: {model_path}")
            return None, None, None

        # preprocessor_config.json 확인 및 생성
        ensure_preprocessor_config_if_missing(model_path)

        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        return model, tokenizer, image_processor
    except Exception as e:
        print(f"[ERROR] Failed to load OCR model from local path: {e}")
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
    # 1) 로컬에서 OCR 모델 로드
    model, tokenizer, image_processor = load_ocr_model(LOCAL_MODEL_PATH)
    if not model or not tokenizer or not image_processor:
        print("[ERROR] OCR 모델 로드에 실패했습니다.")
        return

    # 2) 크롭된 이미지 디렉토리 확인
    if not os.path.exists(CROPPED_IMAGES_DIR):
        print(f"[ERROR] Cropped 이미지 폴더가 없습니다: {CROPPED_IMAGES_DIR}")
        return

    # 3) 자식 폴더 리스트 생성
    subfolders = [
        os.path.join(CROPPED_IMAGES_DIR, folder)
        for folder in os.listdir(CROPPED_IMAGES_DIR)
        if os.path.isdir(os.path.join(CROPPED_IMAGES_DIR, folder))
    ]

    if not subfolders:
        print("[ERROR] 자식 폴더가 없습니다.")
        return

    # 폴더별 OCR 결과 저장
    folder_results = {}

    for subfolder in sorted(subfolders):
        print(f"[INFO] Processing subfolder: {subfolder}")

        # 이미지 파일 리스트 생성 (오름차순 정렬)
        image_paths = [
            os.path.join(subfolder, fname)
            for fname in sorted(os.listdir(subfolder))
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_paths:
            print(f"[WARN] No images found in {subfolder}. Skipping...")
            continue

        # OCR 수행
        folder_texts = perform_ocr_on_cropped_images(
            image_paths=image_paths,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_size=384,
        )

        # 폴더 이름을 키로 결과 저장
        folder_name = os.path.basename(subfolder)
        folder_results[folder_name] = " ".join(folder_texts).strip()  # folder_results: 크롭된 텍스트를 모두 합친 최종 문자열

    # 최종 결과 JSON 생성 및 저장
    final_result = {
        "cropped_images_dir": CROPPED_IMAGES_DIR,
        "ocr_results": folder_results
    }

    # JSON 파일로 저장 (results 폴더)
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_json_path = os.path.join(results_dir, "ocr_results.json")

    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(final_result, json_file, ensure_ascii=False, indent=4)
    print(f"[INFO] OCR results saved to: {output_json_path}")

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

if __name__ == "__main__":
    main()

# LLM 한테 json 형태로 전달: "ocr_results.json"
