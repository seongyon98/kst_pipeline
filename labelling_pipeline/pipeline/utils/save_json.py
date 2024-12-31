import os
import json


# 7. JSON 저장
def save_to_json(image_id, bboxes, question_text, output_path):
    try:
        # 기존 JSON 파일 삭제
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"[INFO] Existing JSON file deleted: {output_path}")

        data = {
            "image_id": image_id,
            "bboxes": bboxes,
            "question_text": " ".join(question_text),
        }
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Saved JSON to {output_path}")
        return output_path
    except Exception as e:
        print(f"[ERROR] Failed to save JSON: {e}")
        raise e
