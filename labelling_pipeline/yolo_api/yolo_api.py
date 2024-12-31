from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
from yolo_model import load_yolo_model_from_s3, extract_bboxes

app = FastAPI()

# YOLO 모델 로드 (S3에서 다운로드)
yolo_model = load_yolo_model_from_s3()


@app.post("/extract_bboxes/")
async def extract_bboxes_from_image(file: UploadFile = File(...)):
    try:
        # 이미지 파일 읽기
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))

        # YOLO 모델을 사용해 바운딩 박스 추출
        bboxes = extract_bboxes(yolo_model, image)

        return JSONResponse(content={"bboxes": bboxes})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
