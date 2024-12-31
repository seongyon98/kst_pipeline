from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
from ocr_model import load_finetuned_trocr_model_from_s3, extract_text_from_bboxes

app = FastAPI()

# OCR 모델 로드 (S3에서 다운로드)
ocr_processor, ocr_model = load_finetuned_trocr_model_from_s3()


@app.post("/extract_text/")
async def extract_text_from_image(file: UploadFile = File(...), bboxes: list = None):
    try:
        if bboxes is None:
            return JSONResponse(
                content={"error": "Bounding boxes are required!"}, status_code=400
            )

        # 이미지 파일 읽기
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))

        # OCR 모델을 사용하여 바운딩 박스 내에서 텍스트 추출
        text = extract_text_from_bboxes(ocr_processor, ocr_model, image, bboxes)

        return JSONResponse(content={"text": text})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
