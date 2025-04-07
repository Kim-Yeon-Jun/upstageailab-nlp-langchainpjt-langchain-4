from typing import Dict, List
from langchain.vectorstores.base import VectorStore
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool

def get_tool_description(category: str) -> str:
    descriptions = {
        "부영_출산정책": "부영그룹이나 저출생 정책에 관한 정보를 검색할 때 사용하세요.",
        "정부정책": "정부 정책이나 제도에 관한 정보를 검색할 때 사용하세요.",
        "스포츠": "스포츠 경기 결과나 관련 정보를 검색할 때 사용하세요.",
        "삼성전자_ai": "삼성전자의 ai 개발이나 정부의 ai 정책 관련 정보를 검색할 때 사용하세요.",
    }
    return descriptions.get(category, f"{category}에 관한 정보를 검색할 때 사용하세요.")

def create_retriever_tools(vectorstores: Dict[str, VectorStore]) -> List[Tool]:
    tools = []
    for category, vectorstore in vectorstores.items():
        retriever = vectorstore.as_retriever()
        tool = create_retriever_tool(
            retriever,
            name=category,
            description=get_tool_description(category)
        )
        tools.append(tool)
    return tools