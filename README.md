[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/5BS4k7bR)
# **LangChain 프로젝트** *(예시)*

LangChain과 Upstage Embedding을 활용하여, 여러 문서(PDF, 크롤링, Python 코드)를 기반으로 대화형 질의응답 시스템을 구축한 프로젝트입니다.
RAG(Retrieval-Augmented Generation) 구조와 LangServe API 구조를 활용해 실시간 질의응답 서비스 형태로 통합했습니다.

- **프로젝트 기간:** 2025.04.02 ~ 2025.04.08  
- **주제:** LangChain 기반 문서 검색 + Q&A 자동화 시스템  

---

# **팀원 소개**

| 이름      | 역할             | GitHub                | 담당 기능                                         |
|-----------|------------------|------------------------|--------------------------------------------------|
| **김패캠** | 팀장 / 역할 | [GitHub 링크](#)       | LangChain 통합, FastAPI 백엔드 구성, API 설계 및 구조화 |
| **박패캠** |  역할   | [GitHub 링크](#)       | CI/CD 파이프라인 구축, 도커화, 로컬/클라우드 환경 구성 |
| **이패캠** | 역할 | [GitHub 링크](#)       | 문서 임베딩 처리, 벡터 DB 구축, LLM 파인튜닝           |
| **최패캠** | 역할     | [GitHub 링크](#)       | 데이터 수집, 전처리, DVC 및 S3 데이터 관리            |

---

# **파이프라인 워크플로우**

LangChain 기반 문서 QA 시스템의 구축 및 운영을 위한 파이프라인입니다.

## **1. 비즈니스 문제 정의**
- 복잡한 문서에서 원하는 정보를 쉽게 찾으 수 있는 자동 질의응답 시스템 필요
- 내부 지식베이스 활용 극대화
- 목표 KPI: 응답 정확도 향상, 사용자 피드백 만족도 확보

## **2. 데이터 수집 및 전처리**
- .pkl 형태로 정제된 문서 데이터 사용
- 카테고리별로 문서를 불러와 Markdown 또는 Python 코드 등 형식에 맞게 청크 분할
- LangChain의 TextSplitter 활용

## **3. 벡터화 및 툴 생성**
- Upstage Embedding 기반으로 FAISS 벡터스토어 생성
- 카테고리별 벡터스토어를 기반으로 LangChain Tool 자동 생성
- DuckDuckGo Tool과 통합하여 최신 정보 검색도 가능

## **4. LLM 및 RAG 파이프라인 구성**
- ChatUpstage 모델 기반 LLM 응답 처리
- LangChain의 Tool Calling Agent 사용
- 사용자 입력과 대화 히스토리를 기반으로 실행기 구성성

## **5. API 제공 및 실행 환경**
- FastAPI + LangServe 조합으로 RAG 응답 API 제공
- /upstage_chain 엔드포인트로 agent 노출
- 로컬 환경 또는 EC2 등 서버에서도 동일하게 구동 가능

## **6. 모니터링 및 재학습 루프**
1. **모델 성능 모니터링**
   - Prometheus, Grafana를 통한 응답 시간 및 정확도 트래킹
2. **데이터 Drift 탐지**
   - Evidently AI 활용
3. **사용자 피드백 루프**
   - 사용자의 thumbs-up/down 기록을 통해 성능 개선
   - 재학습 조건 충족 시 자동 트리거되는 학습 파이프라인 구성

---

## **프로젝트 실행 방법**

본 프로젝트는 웹 서비스 형태로 배포하지 않아도 되며,  
**로컬 환경 또는 클라우드 인스턴스에서 터미널 기반으로 실행** 가능합니다.

```bash
# 1. 프로젝트 클론
git clone https://github.com/your-org/langchain-rag-agent.git
cd langchain-rag-agent

# 2. 가상환경 설정 및 패키지 설치
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. .env 환경 변수 설정
OPENAI_API_KEY=your-api-key
UPSTAGE_API_KEY=your-upstage-key

# 4. FastAPI 서버 실행
uvicorn main:app --reload --port 30032
```

---

## **활용 장비 및 사용 툴**

### **활용 장비**
- **서버:** AWS EC2 (m5.large), S3, ECR
- **개발 환경:** Ubuntu 22.04, Python 3.10+
- **테스트 환경:** NVIDIA V100 GPU 서버 (Lambda Labs 등)

### **협업 툴**
- **소스 관리:** GitHub
- **프로젝트 관리:** GitHub
- **커뮤니케이션:** Slack
- **버전 관리:** Git

### **사용 도구**
- **CI/CD:** GitHub Actions, Jenkins
- **LLM 통합:** LangChain, OpenAI API, HuggingFace
- **실험 관리:** MLflow, Optuna
- **데이터 관리:** DVC, AWS S3
- **모니터링:** Prometheus, Grafana, ELK Stack
- **배포 및 운영:** Docker, Kubernetes, Helm

---

## **기대 효과 및 향후 계획**
- 문서 기반 질문 응답 자동화로 고객 응대 시간 절감
- 사내 문서 검색 정확도 및 사용성 향상
- 향후 다양한 도메인 문서(QA, 정책, 교육자료 등)에 확장 적용 예정

---
## **강사님 피드백 및 프로젝트 회고**

프로젝트 진행 중 담당 강사님과의 피드백 세션을 통해 얻은 주요 인사이트는 다음과 같습니다.

### 📌 **1차 피드백 (2025.04.02)**
- **LangChain, RAG 학습 부족**  
  → 전체적으로 LangChain과 RAG 선행이 되어있지 않아서 일단 강의를 보고 개념을 이해하는 시간을 가지기.

### 📌 **2차 피드백 (2025.04.03)**
- **주제 빨리 정하기**  
- **Native RAG 코드에서 프로젝트 데이터로 바꿔서 결과보기**  
- **금요일(내일)에는 코드 보면서 멘토링 진행 예정**

### 📌 **3차 피드백 (2025.04.04)**
- **기존 NativeRAG에서 발전 가능성 찾아보기**  
  → Advanced RAG 기법 알아보기.
- **GitHub repository 모든 팀원이 Contributor가 되기**  
- **마지막 발표 출력형태는 streamlit 데모 형태**

### 📌 **4차 피드백 (2025.04.07)**
- **코드 Class로 구현하기**  
- **LLM을 통한 프롬프팅으로 의도 분석하는 프로세스 구현하기**  
- **Embedding, retriever 방식 수정해보기**
- **코드 return 형식도 볼 수 있게 코드 깔끔하게 정리하기**
- **GitHub test 코드들은 폴더에 넣어서 정리하기**
- **프롬프트를 풍부하게 작성(순서대로 답변해줘 등)**
- **ReAct 프롬프트 구현(agent 만들 때 사용)**
