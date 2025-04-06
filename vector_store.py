import os
import pickle
import bs4
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.embeddings import OpenAIEmbeddings
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

# 벡터스토어 생성 함수
def create_category_vectorstores(category_docs, save_dir="vectorstores"):
    """카테고리별 벡터스토어를 생성합니다."""
    # os.makedirs(save_dir, exist_ok=True)
    vectorstores = {}
    embeddings = UpstageEmbeddings(model="embedding-query")
    
    for category, docs in category_docs.items():
        if not docs:
            continue
            
        vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
        # save_path = os.path.join(save_dir, category)
        # 한글이름떄문에 저장할때 에러 발생. 일단 주석처리
        # vectorstore.save_local(save_path)
        vectorstores[category] = vectorstore
        
        print(f"{category} 벡터스토어 생성 완료 ({len(docs)}개 문서)")
    
    return vectorstores