import openai
from dotenv import load_dotenv
import os
import time
import boto3
import json
import psycopg2
from datetime import datetime

# 환경 변수 로드
load_dotenv(override=True)

POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# PostgreSQL 연결 설정
DB_CONFIG = {
    "dbname": POSTGRES_DB,
    "user": POSTGRES_USER,
    "password": POSTGRES_PASSWORD,
    "host": POSTGRES_HOST,
    "port": POSTGRES_PORT,
}


# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")


# S3에서 JSON 파일을 로드하여 파싱
def load_json_from_s3(bucket_name, file_path):
    s3 = boto3.client("s3")
    try:
        # S3에서 파일을 읽어오기
        response = s3.get_object(Bucket=bucket_name, Key=file_path)
        content = response["Body"].read().decode("utf-8")

        # 파일 내용이 비어 있지 않은지 확인
        if not content.strip():  # 비어 있으면
            print(f"[WARNING] 파일 '{file_path}'이 비어 있습니다.")
            return None

        return json.loads(content)  # JSON 파싱
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 파싱 오류: {file_path} - {e}")
        return None
    except Exception as e:
        print(f"[ERROR] S3에서 파일 로드 실패: {e}")
        return None


# S3에서 대분류와 일치하는 JSON 데이터를 검색
def search_s3_files_for_category(bucket_name, prefix, target_category):
    s3 = boto3.client("s3")
    try:
        # S3에서 객체 목록을 가져옴
        result = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        for obj in result.get("Contents", []):
            file_path = obj["Key"]

            # 폴더인지 파일인지 확인
            if file_path.endswith("/"):  # 폴더일 경우 건너뜀
                print(f"[INFO] '{file_path}'는 폴더입니다. 건너뜁니다.")
                continue

            # JSON 파일 로드
            json_data = load_json_from_s3(bucket_name, file_path)
            if json_data:
                summary = summarize_json_hierarchy(json_data)
                if target_category in summary:
                    return json_data

        print(
            f"[ERROR] S3 내 파일에서 대분류 '{target_category}'에 해당하는 데이터를 찾을 수 없습니다."
        )
    except Exception as e:
        print(f"[ERROR] S3 파일 검색 중 오류 발생: {e}")
    return None


# JSON 데이터를 계층 구조로 요약하여 문자열로 반환
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


# 수학적 개념 추출
def extract_math_concepts(problem_text):
    """
    문제에서 수학적 개념을 추출합니다.
    """
    prompt = (
        "다음은 한국 초등학교 수학 문제입니다:\n"
        f"{problem_text}\n\n"
        "문제를 분석하여 해결하는 데 필요할 것으로 예상되는 주요 초등 수학적 개념을 추출하세요.\n"
        "길이, 들이, 넓이, 무게, 부피 등의 단위나 도형 및 선분, 직선, 규칙이나 배열, 표나 그래프가 문제에 나오는 경우 해당 개념은 반드시 추출하는 개념에 포함하세요.\n"
        "추출된 주요 개념의 명칭만 작성하세요."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.5,
        )
        math_concept = response.choices[0].message["content"].strip()
        print(f"[INFO] 추출된 수학적 개념: {math_concept}")
        return math_concept
    except Exception as e:
        print(f"[ERROR] 수학적 개념 추출 실패: {e}")
        return None


# 대분류 추출
def determine_major_category(math_concept):
    """
    수학적 개념을 분석하여 적합한 대분류를 결정합니다.
    """
    prompt = (
        f"다음은 추출된 수학적 개념입니다:\n{math_concept}\n\n"
        "이 개념과 가장 유사한 대분류를 '수와 연산', '변화와 관계', '도형과 측정', '자료와 가능성' 중에서 선택하세요. "
        "단, 다음 규칙을 따라 우선적으로 선택하세요:\n"
        "- 그래프나 표와 관련된 문제는 '자료와 가능성'\n"
        "- 길이, 넓이, 들이, 무게 등의 단위, 도형, 각도, 선분 및 직선은 '도형과 측정'\n"
        "- 비례, 대응, 비교, 배열, 규칙은 '변화와 관계'\n"
        "- 사칙연산(덧셈, 뺄셈, 곱셈, 나눗셈)은 '수와 연산'\n"
        "어디에도 속하지 않는 경우 '수와 연산'으로 분류하세요."
        "대분류명만 문장 부호를 제외하고 작성하세요."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.5,
        )
        major_category = response.choices[0].message["content"].strip()
        print(f"[INFO] 결정된 대분류: {major_category}")
        return major_category
    except Exception as e:
        print(f"[ERROR] 대분류 결정 실패: {e}")
        return None


# 최하위분류 추출
def extract_leaf_category_within_major_category(
    math_concept, major_category, json_summary
):
    """
    JSON 데이터 요약을 기반으로, 특정 대분류 내에서 최하위 분류를 추출합니다.
    """
    prompt = (
        f"다음은 추출된 수학적 개념입니다:\n{math_concept}\n\n"
        f"대분류: {major_category}\n\n"
        "다음은 학습 주제의 계층 구조입니다. 이 계층 구조에서 해당 대분류에 속하는 내용만 사용하여 최하위 분류를 추출하세요:\n"
        f"{json_summary}\n\n"
        "추출된 개념에 가장 유사한 최하위 분류(학습 주제)를 계층 구조에서 추출하세요. "
        "단, 자식이 없는 최하위 분류만 추출하며, 가능한 한 1개만 추출하세요. "
        "최하위 분류가 불가피하게 여러 개 필요한 경우 최대 2개까지만 콤마로 구분하여 추출하세요.\n\n"
        "최하위 분류명만 문장 부호를 제외하고 작성하세요."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.5,
        )
        leaf_category = response.choices[0].message["content"].strip()
        print(f"[INFO] 대분류 '{major_category}' 내 최하위 분류: {leaf_category}")
        return leaf_category
    except Exception as e:
        print(f"[ERROR] 최하위 분류 추출 실패: {e}")
        return None


# 문제 데이터 처리
def process_math_problem(
    problem_text,
    bucket_name,
    category_map,
    prefix,
    figure_text=None,
    model="gpt-4-turbo",
):
    """
    문제를 분석하여 대분류 및 최하위 분류를 추출합니다.
    figure_text가 주어지면 해당 텍스트도 함께 고려하여 분석합니다.

    Parameters:
    - problem_text (str): 수학 문제 텍스트
    - bucket_name (str): S3 버킷 이름
    - category_map (dict): 대분류와 S3 파일 매핑
    - prefix (str): S3 경로 접두사
    - figure_text (str, optional): 문제와 관련된 그림 설명 텍스트
    - model (str): GPT 모델 이름 (기본값: "gpt-4")

    Returns:
    - tuple: (대분류, 최하위 분류, 대분류 추출 시간, 최하위 분류 추출 시간)
    """
    # Step 1: 문제 분석을 통해 주요 개념 및 대분류 추출
    start_time = time.time()

    # 문제 텍스트와 figure_text를 합쳐서 분석하도록 수정
    combined_text = problem_text
    if figure_text:
        combined_text += "\n" + figure_text  # figure_text가 있으면 추가하여 처리

    # 수학적 개념 추출
    math_concept = extract_math_concepts(combined_text)
    major_category = determine_major_category(math_concept)
    category_time = time.time() - start_time
    print(f"[INFO] 추출된 대분류: {major_category}")

    # Step 2: S3에서 대분류에 해당하는 JSON 파일을 검색하여 최하위 분류 정보 추출
    leaf_category_data = None
    leaf_time = 0  # 최하위 분류 추출에 걸린 시간

    if major_category:
        try:
            # S3에서 대분류에 해당하는 파일을 검색
            start_leaf_time = time.time()
            json_data = search_s3_files_for_category(
                bucket_name, prefix, major_category
            )

            if json_data:
                # 최하위 분류 추출
                leaf_category_data = extract_leaf_category_within_major_category(
                    math_concept, major_category, json_data
                )
                leaf_time = time.time() - start_leaf_time

                if leaf_category_data:
                    print(f"[INFO] 최하위 분류: {leaf_category_data}")
                else:
                    print(
                        f"[ERROR] '{major_category}' 대분류 내 최하위 분류 데이터를 찾을 수 없습니다."
                    )
            else:
                print(
                    f"[ERROR] '{major_category}' 대분류에 해당하는 JSON 데이터를 찾을 수 없습니다."
                )
        except Exception as e:
            print(f"[ERROR] 최하위 분류 추출 중 오류 발생: {e}")

    # Step 3: 결과 반환
    return major_category, leaf_category_data, category_time, leaf_time


def save_file_record(file_name, s3_key):
    """upload_filerecord에 파일 메타데이터를 저장"""
    try:
        print(f"[DEBUG] 파일 메타데이터 저장 중... (파일: {file_name})")

        # PostgreSQL 연결
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # upload_filerecord 테이블에 데이터 삽입
        cur.execute(
            """
            INSERT INTO upload_filerecord (file_name, s3_key, status, uploaded_at)
            VALUES (%s, %s, %s, %s) RETURNING id;
            """,
            (file_name, s3_key, "complete", datetime.now()),
        )

        # 삽입된 file_record id 가져오기
        file_record_id = cur.fetchone()[0]

        conn.commit()

        print(f"[INFO] 파일 메타데이터 저장 성공: {file_name}, id: {file_record_id}")

        cur.close()
        conn.close()

        return file_record_id  # 저장된 file_record_id를 반환

    except Exception as e:
        print(f"[ERROR] 파일 메타데이터 저장 실패: {e}")
        raise e  # 에러를 상위 호출 함수로 전달


def save_to_db(file_name, s3_key, category, leaf_category):
    """결과를 PostgreSQL에 저장 (psycopg2 사용)"""
    try:
        print(
            f"[DEBUG] 데이터베이스에 저장 중... (파일: {file_name}, 대분류: {category}, 최하위분류: {leaf_category})"
        )

        # 파일 메타데이터 저장
        file_record_id = save_file_record(file_name, s3_key)

        # PostgreSQL 연결
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # LabellingResult 생성
        cur.execute(
            """ 
            INSERT INTO upload_labellingresult (file_record_id, category, leaf_category, processed_at)
            VALUES (%s, %s, %s, %s);
            """,
            (file_record_id, category, leaf_category, datetime.now()),
        )

        conn.commit()

        print(
            f"[INFO] 저장 성공: {file_name}, 라벨링 결과: 대분류: {category}, 최하위 분류: {leaf_category}"
        )

        cur.close()
        conn.close()

    except Exception as e:
        print(f"[ERROR] 저장 실패: {e}")
        raise e  # 에러를 상위 호출 함수로 전달


# ocr까지 해서 전처리된 텍스트를 바로 넘겨받는다고 가정
def process_multiple_problems(file_name, s3_key, question_text):
    # JSON 파일 (로드맵 파일) 관련 정보
    json_bucket_name = "big9-project-02-roadmap-bucket"  # JSON 파일이 위치한 S3 버킷
    json_prefix = "roadmap_2022/"  # JSON 파일이 위치한 S3 폴더
    category_map = {
        "수와 연산": "01_num_cal.json",
        "변화와 관계": "02_change_of_relationship.json",
        "도형과 측정": "03_shape_meas.json",
        "자료와 가능성": "04_data_and_possibility.json",
    }

    if not question_text:
        raise ValueError("문제 텍스트가 비어있습니다.")

    print(f"문제 텍스트: {question_text}")

    try:
        # 문제 처리
        category, leaf_category, category_time, leaf_time = process_math_problem(
            problem_text=question_text,
            bucket_name=json_bucket_name,
            category_map=category_map,
            prefix=json_prefix,
            model="gpt-4-turbo",  # 기본 모델을 "gpt-4o"로 설정
        )

        spent_time = category_time + leaf_time

        print(
            f"[DEBUG] 처리 결과: 대분류: {category}, 최하위 분류: {leaf_category}, 처리 시간: {spent_time}"
        )

        # PostgreSQL에 저장
        save_to_db(file_name, s3_key, category, leaf_category)

        return category, leaf_category, category_time, leaf_time

    except Exception as e:
        print(f"[ERROR] 문제 처리 실패: {e}")
        raise e  # 예외를 상위 호출 함수로 전달
