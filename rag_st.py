import streamlit as st
from streamlit_chat import message
from langchain_core.messages import HumanMessage, AIMessage
from langserve import RemoteRunnable
import pandas as pd
import requests
import json
import uuid


class rag_web():
    def __init__(self):
        self.BASE_URL = "http://localhost:30032"
        self.URL_DICT = {
            "LangChain": f"{self.BASE_URL}/upstage_chain/"
        }
        # 세션 ID 초기화
        if "session_id" not in st.session_state:
            st.session_state.session_id = f"user-session-{uuid.uuid4()}"
        
        # 채팅 기록 초기화
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        self.gen_sidebar()

    def gen_sidebar(self):
        with st.sidebar:
            st.radio(
                "select the model to use!",
                ["LangChain"],
                key='model',
                on_change=self.rest_log
            )
            # 세션 ID 표시 (디버깅용, 필요없으면 제거)
            st.text(f"Session ID: {st.session_state.session_id}")
            if st.button("New Session"):
                st.session_state.session_id = f"user-session-{uuid.uuid4()}"
                self.rest_log()
                st.experimental_rerun()

    def get_answer_langserve(self, message):
        url = self.URL_DICT[st.session_state['model']]
        lang = RemoteRunnable(url)
        
        # RunnableWithMessageHistory에 맞는 입력 형식 구성
        input_data = {
            "input": message,  # 단일 input 문자열
            "config": {
                "configurable": {
                    "session_id": st.session_state.session_id
                }
            }
        }
        print(f"Input data: {input_data}")  # 디버깅용
        print(f"session_id: {st.session_state.session_id}")  # 디버깅용

        try:
            result = lang.invoke(input_data)
            
            # 응답 결과 처리
            if isinstance(result, dict) and "output" in result:
                response = result["output"]
            else:
                response = str(result)
                
            return_image = False
            source_nodes = []
        
            
            # 대화 기록 업데이트 (메시지 객체는 서버에서 관리)
            return response, source_nodes, return_image
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return f"Error: {str(e)}", [], False
    
    def get_answer(self, message):
        return self.get_answer_langserve(message)

    def window(self):
        st.title("My RAG Model")
        st.subheader("langchain API")
        st.caption("created by team4")

        st.text_input("type a message..", key="user_message")
        if user_message := st.session_state.user_message:
            response, source_nodes, return_image = self.get_answer(user_message)
            st.success(response)

            source_nodes = pd.DataFrame(source_nodes)
            if "node" in source_nodes.columns:
                source_nodes[['text', 'metadata']] = source_nodes['node'].apply(lambda x: pd.Series((x['text'], x['metadata']['file_name'])))
                source_nodes = source_nodes.drop('node', axis=1)
            st.dataframe(source_nodes)
    
    def update_log(self, user_message, bot_message):
        if "chat_log" not in st.session_state:
            st.session_state.chat_log = {"user_message": [], "bot_message":[]}

        st.session_state.chat_log["user_message"].append(user_message)
        st.session_state.chat_log["bot_message"].append(bot_message)

    def display_chat_log(self):
        if "chat_log" not in st.session_state:
            return
            
        bot_messages = st.session_state.chat_log["bot_message"][::-1]
        user_messages = st.session_state.chat_log["user_message"][::-1]

        for idx, (bot, user) in enumerate(zip(bot_messages, user_messages)):
            message(bot, key=f"{idx}_bot")
            message(user, key=str(idx), is_user=True)
    
    def rest_log(self):
        st.session_state.chat_log = {"user_message": [], "bot_message":[]}
        st.session_state.bot_output = None, None, None

    def user_input(self):
        user_message = st.session_state.user_message
        st.session_state['user_message'] = ""
        st.session_state['bot_output'] = self.get_answer(user_message)
        self.update_log(user_message, st.session_state['bot_output'][0])


    def window_chat(self):
        st.title("My RAG Model")
        st.subheader("langchain API")
        st.caption("created by team4")

        st.text_input("type a message..", key="user_message", on_change=self.user_input)
        if st.button("reset"):
            self.rest_log()
            self.display_chat_log()

        if 'bot_output' in st.session_state:
            response, source_nodes, return_image = st.session_state['bot_output']
            self.display_chat_log()

            # 소스 노드 표시 (필요한 경우 주석 해제)
            # source_nodes = pd.DataFrame(source_nodes)
            # if "node" in source_nodes.columns:
            #     source_nodes[['text', 'metadata']] = source_nodes['node'].apply(lambda x: pd.Series((x['text'], x['metadata']['file_name'])))
            #     source_nodes = source_nodes.drop('node', axis=1)
            # st.dataframe(source_nodes)


if __name__ == "__main__":
    st.set_page_config(
        page_title="RAG web",
        page_icon="🧊",
    )
    web = rag_web()
    web.window_chat()