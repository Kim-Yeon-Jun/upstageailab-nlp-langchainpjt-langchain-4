from typing import Dict
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools import Tool
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.tools import DuckDuckGoSearchRun
from utils.session import get_session_history
from split_load import load_all_data
from vector_store import create_category_vectorstores
from create_tools import create_retriever_tools

class AgentManager:
    def __init__(self):
        load_dotenv()
        self.store: Dict[str, any] = {}

        self.embeddings = UpstageEmbeddings(model="embedding-query")
        self.llm = ChatUpstage(model="solar-mini")
        self.search_tool = Tool(
            name="웹_검색",
            func=DuckDuckGoSearchRun().run,
            description="웹에서 최신 정보를 검색할 때 사용하세요. 내부 지식 베이스에서 찾을 수 없는 정보를 검색하는 데 유용합니다."
        )

        category_docs, _ = load_all_data()
        vectorstores = create_category_vectorstores(category_docs)
        self.tools = create_retriever_tools(vectorstores) + [self.search_tool]

        self.prompt = ChatPromptTemplate.from_messages([
            (
            "system",
            "당신은 유용한 도우미 AI입니다."
            "기업, 정책, 파이썬 코드 등 특정 주제에 관한 질문이라면 해당 카테고리의 도구를 사용하세요."
            "내부 지식 베이스에서 정보를 찾을 수 없다면, `웹_검색` 도구를 사용하여 웹에서 정보를 검색하세요."
        ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        self.agent_executor = AgentExecutor(
            agent=create_tool_calling_agent(self.llm, self.tools, self.prompt),
            tools=self.tools,
            verbose=False
        )

    def get_agent(self) -> RunnableWithMessageHistory:
        return RunnableWithMessageHistory(
            self.agent_executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )