from langchain.vectorstores import FAISS
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
 
from langchain_upstage import UpstageEmbeddings
from langchain_upstage import ChatUpstage

# 리트리버 도구 생성 함수
def create_retriever_tools(vectorstores):
    """카테고리별 리트리버 도구를 생성합니다."""
    tools = []

    for category, vectorstore in vectorstores.items():
        # retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        retriever = vectorstore.as_retriever()
        
        # 카테고리별 맞춤 설명 생성
        if category == "부영_출산정책":
            description = "부영그룹이나 저출생 정책에 관한 정보를 검색할 때 사용하세요."
        elif category == "정부정책":
            description = "정부 정책이나 제도에 관한 정보를 검색할 때 사용하세요."
        elif category == "스포츠":
            description = "스포츠 경기 결과나 관련 정보를 검색할 때 사용하세요."
        elif category == "삼성전자_ai":
            description = "삼성전자의 ai 개발이나 정부의 ai 정책 관련 정보를 검색할 때 사용하세요."
        else:
            description = f"{category}에 관한 정보를 검색할 때 사용하세요."
        
        tool = create_retriever_tool(
            retriever,
            name=f"{category}",
            description=description
        )
        tools.append(tool)
    
    return tools