import openai
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# .env 파일 로드
load_dotenv()

# 환경 변수에서 값 가져오기
openai.api_key = os.getenv("OPENAI_API_KEY")
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

# Neo4j 데이터베이스 연결
driver = GraphDatabase.driver(uri, auth=(username, password))
session = driver.session()


# 키워드 전처리 함수
def preprocess_keywords(keywords):
    processed = [kw.strip().lower() for kw in keywords.split(",") if kw.strip()]
    return processed


# LLM을 사용한 수학 키워드 및 대분류 추출 함수
def extract_keywords_and_category_from_math_problem(text):
    prompt = (
        "다음 수학 문제에서 수학과 관련된 키워드와 가장 적합한 대분류를 추출하세요. "
        "대분류는 '수와 연산', '변화와 관계', '도형과 측정', '자료와 가능성' 중 하나이어야 합니다.\n"
        f"문제: {text}"
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.5,
    )
    return response.choices[0].message["content"].strip()


# 자카드 유사도 계산 함수
def jaccard_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    return float(len(set1 & set2)) / len(set1 | set2)


# 하위 노드만 찾는 함수
def find_similar_leaf_nodes_from_graph(keywords, category):
    query = """
    MATCH (c:Category {name: $category})-[:HAS_CHILD*]->(i:Item)
    WHERE NOT (i)-[:HAS_CHILD]->()  // 최하위 노드만 선택
    RETURN i.name AS item
    """
    results = session.run(query, category=category)
    leaf_nodes = [record["item"].lower() for record in results]

    matched_items = []

    for item_name in leaf_nodes:
        for keyword in keywords:
            similarity = jaccard_similarity(keyword, item_name)
            print(
                f"키워드 '{keyword}' vs 데이터베이스 항목 '{item_name}': 유사도 {similarity:.2f}"
            )  # 디버깅용
            if similarity > 0.2:  # 유사도 기준 20%
                matched_items.append(item_name)

    return list(set(matched_items))  # 중복 제거


# 분류 검증하는 LLM 함수
def verify_classification_with_llm(classification, original_problem):
    prompt = f"수학 문제 '{original_problem}'이 분류 '{classification}'에 적합한지 확인하고, 이유를 설명해주세요."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=100,
        temperature=0.5,
    )
    return response.choices[0].message["content"].strip()


# 전체 파이프라인 실행 함수
def process_math_problem(problem):
    # 1단계: 키워드 및 대분류 추출
    raw_keywords_and_category = extract_keywords_and_category_from_math_problem(problem)
    print(f"추출된 키워드 및 대분류 (원본): {raw_keywords_and_category}")

    # 키워드와 대분류 분리
    raw_keywords, most_similar_category = raw_keywords_and_category.rsplit("\n", 1)
    keywords = preprocess_keywords(raw_keywords)
    print(f"전처리된 키워드: {keywords}")
    print(f"가장 유사한 대분류: {most_similar_category}")

    # 2단계: 하위 노드에서 유사한 항목 찾기
    matched_items = find_similar_leaf_nodes_from_graph(keywords, most_similar_category)
    print(f"찾은 하위 노드 항목: {matched_items}")

    # 3단계: 항목 검증
    for item in matched_items:
        verification = verify_classification_with_llm(item, problem)
        print(f"항목 '{item}' 검증 결과: {verification}")


# 예시 수학 문제
problem = "다음 그림을 보고 어떤 삼각형인지 분류하시오."
process_math_problem(problem)

# Neo4j 세션 종료
session.close()
