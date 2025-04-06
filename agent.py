# API 키를 환경변수로 관리하기 위한 설정 파일
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings, ChatUpstage

from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults

from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_teddynote.messages import AgentStreamParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_teddynote import logging


from dotenv import load_dotenv
from split_load import load_all_data
from vector_store import create_category_vectorstores
from create_tools import create_retriever_tools
# API 키 정보 로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("Agents")

search = DuckDuckGoSearchRun()

embeddings = UpstageEmbeddings(model="embedding-query")
llm = ChatUpstage(model="solar-mini")

category_docs, all_docs = load_all_data()
vectorstore = create_category_vectorstores(category_docs)
tools = create_retriever_tools(vectorstore)

web_search_tool = Tool(
    name="웹_검색",
    func=search.run,
    description="웹에서 최신 정보를 검색할 때 사용하세요. 내부 지식 베이스에서 찾을 수 없는 정보를 검색하는 데 유용합니다."
)

all_tools = tools + [web_search_tool]
# Prompt 정의
prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        "당신은 유용한 도우미 AI입니다."
        "기업, 정책, 파이썬 코드 등 특정 주제에 관한 질문이라면 해당 카테고리의 도구를 사용하세요."
        "내부 지식 베이스에서 정보를 찾을 수 없다면, `웹_검색` 도구를 사용하여 웹에서 정보를 검색하세요."
        # "부영그룹에 대한 정보나 저출생 대책에 관한 정보를 검색할 때는 `부영_출산정책` 도구를 사용하세요. "
        # "삼성전자의 ai 개발 관련 정보를 검색할 때는 `삼성_ai_정책` 도구를 사용하세요. "
        # "만약 뉴스에서 정보를 찾을 수 없다면, `웹_검색` 도구를 사용하여 웹에서 정보를 검색하세요."
        # "내부 지식 베이스에서 정보를 찾을 수 없다면, `웹_검색` 도구를 사용하여 웹에서 정보를 검색하세요."
        # "도구를 사용할 수 있는 경우 항상 적절한 도구를 사용하려고 시도하세요."
        # "당신은 절대 자신이 모르는 정보를 추론하지 않습니다. 도구를 사용해 사실 기반으로 답하세요."
        # "부영그룹, 출산정책, 정부정책, 기업 등 특정 주제에 관한 질문이라면 해당 카테고리의 도구를 사용하세요."
        # "내부 지식 베이스에서 정보를 찾을 수 없다면, `웹_검색` 도구를 사용하여 웹에서 정보를 검색하세요."
        # "도구를 사용할 때는 질문의 맥락과 필요한 정보를 고려하여 적절한 도구를 선택하세요."
        # "정보의 출처를 명확히 하고, 발견한 정보를 바탕으로 종합적인 답변을 제공하세요."
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# tool calling agent 생성
agent = create_tool_calling_agent(llm, all_tools, prompt)
# AgentExecutor 생성
agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=False)

# 각 단계별 출력을 위한 파서 생성
agent_stream_parser = AgentStreamParser()

# session_id 를 저장할 딕셔너리 생성
store = {}

# session_id 를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in store:  # session_id 가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

# 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # 대화 session_id
    get_session_history,
    # 프롬프트의 질문이 입력되는 key: "input"
    input_messages_key="input",
    # 프롬프트의 메시지가 입력되는 key: "chat_history"
    history_messages_key="chat_history",
)