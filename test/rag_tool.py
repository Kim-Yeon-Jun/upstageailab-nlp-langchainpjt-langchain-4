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

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_teddynote.messages import AgentStreamParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()


search = DuckDuckGoSearchRun()

bs4.SoupStrainer(
    "div",
    attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
)

# 뉴스기사 내용을 로드하고, 청크로 나누고, 인덱싱합니다.
loader = WebBaseLoader(
    web_paths=("https://n.news.naver.com/article/437/0000378416",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
)

docs = loader.load()
print(f"문서의 수: {len(docs)}")
docs

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

splits = text_splitter.split_documents(docs)
len(splits)

embeddings = UpstageEmbeddings(model="embedding-query")
llm = ChatUpstage(model="solar-mini")

# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

# 뉴스에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    name="부영_출산정책",  # 도구의 이름을 입력합니다.
    # description="use this tool to search information from the news",  # 도구에 대한 설명을 자세히 기입해야 합니다!!
    description="부영그룹이나 저출생 정책에 관해 검색할때는 이 tool을 이용하세요",  # 도구에 대한 설명을 자세히 기입해야 합니다!!
)

# tools 리스트에 search와 retriever_tool을 추가합니다.
tools = [search, retriever_tool]
# tools = [search]


# Prompt 정의
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            # "You are a helpful assistant. "
            "당신은 유용한 도우미입니다. "
            # "Make sure to use the `news_search` tool for searching information from the news. "
            "부영그룹에 대한 정보나 저출생 대책에 관한 정보를 검색할 때는 `부영_출산정책` 도구를 사용하세요. "
            "만약 뉴스에서 정보를 찾을 수 없다면, `duckduckgo_search` 도구를 사용하여 웹에서 정보를 검색하세요.",
            # "If you can't find the information from the news, use the `search` tool for searching information from the web.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


# tool calling agent 생성
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutor 생성
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

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