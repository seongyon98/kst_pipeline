FROM ultralytics/ultralytics:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_yolo.py .

CMD ["python", "train_yolo.py"]
