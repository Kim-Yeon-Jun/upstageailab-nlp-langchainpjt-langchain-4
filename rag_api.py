from fastapi import FastAPI
from langserve import add_routes

from langchain_api_web import chain
from rag_tool import agent_with_chat_history

app = FastAPI()

@app.get("/")
async def root():
    return {"message":"Hello world"}

# add_routes(app, chain, path="/upstage_chain")
add_routes(app, agent_with_chat_history, path="/upstage_chain")


