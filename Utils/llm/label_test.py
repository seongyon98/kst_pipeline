import openai
import boto3
import json
import os


# S3에서 JSON 파일 로드
def load_json_from_s3(bucket_name, file_path):
    s3 = boto3.client("s3")
    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_path)
        content = response["Body"].read().decode("utf-8")
        return json.loads(content)
    except Exception as e:
        print(f"[ERROR] S3에서 파일 로드 실패: {e}")
        return None


# LLM을 사용해 대분류 추출
def extract_category_from_problem(problem_text):
    prompt = (
        "다음 수학 문제에서 가장 적합한 대분류를 추출하세요. "
        "대분류는 '변화와 관계', '도형과 측정', '자료와 가능성' 중 하나를 먼저 확인하고, "
        "그래프나 표와 관련된 문제는 우선적으로 '자료와 가능성'으로 분류하고, "
        "길이, 넓이, 들이, 무게 등의 단위 관련 문제는 '도형과 측정'으로 분류하며, "
        "배열, 규칙, 비율과 관련된 문제는 '변화와 관계'로 분류하세요. "
        "어디에도 속하지 않으면 '수와 연산'으로 분류하세요. "
        f"문제: {problem_text}\n"
        "대분류:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
            temperature=0.5,
        )
        category = response.choices[0].message["content"].strip()
        return category
    except Exception as e:
        print(f"[ERROR] LLM 추출 실패: {e}")
        return None


# LLM을 사용한 최하위 분류 추출 함수
def extract_leaf_category_from_problem(text, json_data):
    # JSON 데이터를 탐색 가능한 문자열로 변환
    json_summary = summarize_json_hierarchy(json_data)

    prompt = (
        "다음은 특정 대분류에 속하는 수학 학습 주제 계층 구조입니다:\n"
        f"{json_summary}\n\n"
        "그리고 다음은 문제입니다:\n"
        f"{text}\n\n"
        "이 문제에 가장 적합한 최하위 분류(학습 주제)를 반환하세요."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        temperature=0.5,
    )
    return response.choices[0].message["content"].strip()


# JSON 계층 구조 요약 함수
def summarize_json_hierarchy(data, level=1):
    summary = ""
    indent = "  " * level
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict) and "name" in value:
                summary += f"{indent}- {value['name']}\n"
                if "children" in value:
                    summary += summarize_json_hierarchy(value["children"], level + 1)
            elif isinstance(value, dict):
                summary += summarize_json_hierarchy(value, level + 1)
            else:
                summary += f"{indent}- {value}\n"
    return summary


# S3에서 JSON 파일 다운로드
def download_file_from_s3(bucket_name, s3_key, local_path):
    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket_name, s3_key, local_path)
        print(f"[INFO] S3에서 {s3_key}를 {local_path}에 다운로드했습니다.")
    except Exception as e:
        print(f"[ERROR] S3 파일 다운로드 실패: {e}")


# JSON 데이터 로드
def load_json_data(local_path):
    try:
        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[ERROR] JSON 파일 로드 실패: {e}")
        return None


# 대분류와 파일 매핑
def get_file_path_from_category(category):
    category_file_map = {
        "수와 연산": "01_num_cal.json",
        "변화와 관계": "02_change_of_relationship.json",
        "도형과 측정": "03_shape_meas.json",
        "자료와 가능성": "04_data_and_possibility.json",
    }
    bucket_name = "big9-project-02-roadmap-bucket"
    s3_base_path = "roadmap_2022/"

    file_name = category_file_map.get(category)
    if file_name:
        s3_key = os.path.join(s3_base_path, file_name)
        local_path = os.path.join("/tmp", file_name)
        return bucket_name, s3_key, local_path
    else:
        print(f"[ERROR] 대분류 '{category}'에 해당하는 파일 경로를 찾을 수 없습니다.")
        return None, None, None


# 문제 처리
def process_math_problem(problem):
    # 1단계: 문제에서 대분류 추출
    category = extract_category_from_problem(problem)
    print(f"[INFO] 추출된 대분류: {category}")

    # 2단계: 대분류에 해당하는 파일 가져오기
    bucket_name, s3_key, local_path = get_file_path_from_category(category)
    if not bucket_name or not s3_key or not local_path:
        return

    # 3단계: S3에서 파일 다운로드
    download_file_from_s3(bucket_name, s3_key, local_path)

    # 4단계: JSON 데이터 로드
    json_data = load_json_data(local_path)
    if not json_data:
        return

    # 5단계: 최하위 분류 추출
    leaf_category = extract_leaf_category_from_problem(problem, json_data)
    if leaf_category:
        print(f"[INFO] 문제에 적합한 최하위 분류: {leaf_category}")
    else:
        print("[INFO] 적합한 최하위 분류를 찾지 못했습니다.")


problem = "기차에 1500명이 타있을 때 24명이 내리고 156명이 타면 기차에는 현재 몇 명이 타있을까요?"
process_math_problem(problem)
