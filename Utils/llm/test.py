import openai
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever

# .env 파일 로드
load_dotenv()

# 환경 변수에서 값 가져오기
openai.api_key = os.getenv("OPENAI_API_KEY")
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
INDEX_NAME = "vector-index-name"

# Neo4j 데이터베이스 연결
driver = GraphDatabase.driver(uri, auth=(username, password))
session = driver.session()

# GraphRAG 설정
embedder = OpenAIEmbeddings(model="text-embedding-3-large")
retriever = VectorRetriever(driver, INDEX_NAME, embedder)
llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})
rag = GraphRAG(retriever=retriever, llm=llm)


# LLM을 사용한 수학 키워드 및 대분류 추출 함수
def extract_keywords_and_category_from_math_problem(text):
    prompt = (
        "다음 수학 문제에서 수학과 관련된 키워드와 가장 적합한 대분류를 추출하세요. "
        "키워드는 '키워드:'로 시작하고 대분류는 '대분류:'로 시작하여 각각 추출해주세요. "
        "대분류는 '변화와 관계', '도형과 측정', '자료와 가능성' 중에서 하나를 먼저 확인하고, "
        "그래프나 표와 관련된 문제는 우선적으로 '자료와 가능성'으로 분류하고, "
        "길이, 넓이, 들이, 무게 등의 단위 관련 문제는 '도형과 측정'으로 분류하며, "
        "배열, 규칙, 비율과 관련된 문제는 '변화와 관계'로 분류하세요. "
        "어디에도 속하지 않으면 '수와 연산'으로 분류하세요.\n"
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


# 키워드 전처리 함수
def preprocess_keywords(raw_keywords):
    # 키워드 부분만 추출
    keywords = raw_keywords.split("키워드:")[1].strip().split(", ")
    # 각 키워드를 소문자로 변환하여 리스트로 반환
    return [kw.lower() for kw in keywords]


# 시드 고정
def set_seed(seed=12345):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 시드 고정 설정
set_seed()


def find_similar_nodes_from_graph(keywords, category, level, similarity_threshold=0.8):
    """
    특정 카테고리 내에서 레벨별로 키워드와 유사도가 높은 항목을 탐색.
    :param keywords: 키워드 리스트
    :param category: 대분류/중분류 이름
    :param level: 검색할 노드의 레벨 ("Subcategory", "Leaf" 등)
    :param similarity_threshold: 유사도 임계값
    :return: 유사도가 높은 노드 리스트
    """
    # GraphRAG를 사용하여 유사도 검색
    query_text = f"Find nodes related to {', '.join(keywords)} in category {category} at level {level}"
    response = rag.search(query_text=query_text, retriever_config={"top_k": 5})

    # 결과에서 유사도 높은 노드들을 필터링
    matched_nodes = [
        node for node in response.answer if node["similarity"] >= similarity_threshold
    ]

    return [node["name"] for node in matched_nodes]


def find_subcategories(keywords, category):
    """
    대분류(category)에서 가장 유사한 중분류 하나를 찾는다.
    :param keywords: 키워드 리스트
    :param category: 대분류 이름
    :return: 가장 유사한 중분류
    """
    best_match = find_similar_nodes_from_graph(keywords, category, level="Subcategory")
    if best_match:
        return best_match

    print(f"중분류를 찾을 수 없습니다. 대분류: {category}")
    return []


def find_leaf_nodes(keywords, category_name, similarity_threshold=0.8):
    # 중분류 찾기
    subcategories = find_subcategories(keywords, category_name)
    if not subcategories:
        print(f"중분류를 찾을 수 없습니다. 대분류: {category_name}")
        return []

    leaf_nodes = []
    for subcategory in subcategories:
        print(f"중분류 '{subcategory}'에서 하위 노드 탐색 시작")
        # 해당 중분류에 연결된 item 찾기
        items = find_similar_nodes_from_graph(
            keywords,
            subcategory,
            level="Item",
            similarity_threshold=similarity_threshold,
        )
        if not items:
            print(f"중분류 '{subcategory}'에 연결된 item이 없습니다.")
            continue
        print(f"중분류 '{subcategory}'에 연결된 item: {items}")

        for item in items:
            # 해당 item에 연결된 leaf 찾기
            query = """
            MATCH (i:Item {name: $item})-[:HAS_CHILD]->(ln:Leaf)
            RETURN ln.name AS leaf
            """
            result = session.run(query, item=item)
            leaf_candidates = [record["leaf"] for record in result]

            for leaf in leaf_candidates:
                # 최하위 노드(leaf) 유사도 검증
                leaf_similarity = calculate_similarity(keywords, leaf)
                if leaf_similarity >= similarity_threshold:
                    print(
                        f"최하위 노드 '{leaf}'는 유사도 {leaf_similarity:.2f}로 선택됨"
                    )
                    leaf_nodes.append(leaf)
                else:
                    print(
                        f"최하위 노드 '{leaf}'는 유사도 {leaf_similarity:.2f}로 선택되지 않음"
                    )

    print(f"최종 선택된 하위 노드(leaf nodes): {leaf_nodes}")
    return list(set(leaf_nodes))


def process_math_problem(problem):
    # 1단계: 키워드 및 대분류 추출
    raw_keywords_and_category = extract_keywords_and_category_from_math_problem(problem)
    print(f"추출된 키워드 및 대분류 (원본): {raw_keywords_and_category}")

    # 키워드와 대분류 분리
    try:
        raw_keywords, most_similar_category = raw_keywords_and_category.split(
            "\n대분류: "
        )
    except ValueError:
        print("응답 형식이 맞지 않습니다. 기본값으로 '수와 연산'을 사용합니다.")
        raw_keywords = raw_keywords_and_category
        most_similar_category = "수와 연산"

    keywords = preprocess_keywords(raw_keywords)
    print(f"전처리된 키워드: {keywords}")
    print(f"가장 유사한 대분류: {most_similar_category}")

    # 2단계: 하위 노드 탐색
    leaf_nodes = find_leaf_nodes(
        keywords, most_similar_category, similarity_threshold=0.6
    )
    if not leaf_nodes:
        print("하위 노드를 찾을 수 없습니다. 프로세스를 종료합니다.")
        return

    print(f"찾은 하위 노드: {leaf_nodes}")

    # 3단계: 최종 검증
    verified_labels = []
    for leaf in leaf_nodes:
        verification = verify_classification_with_llm(leaf, keywords)
        if verification == "적합":
            verified_labels.append(leaf)

    if verified_labels:
        print(f"최종 라벨: {verified_labels}")
    else:
        print("적합한 라벨을 찾을 수 없습니다.")


# Neo4j 세션 종료
session.close()
