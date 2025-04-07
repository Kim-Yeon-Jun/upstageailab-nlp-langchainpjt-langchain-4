from fastapi import FastAPI
from langserve import add_routes
from agent import AgentManager

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello world"}

# 에이전트 라우팅 연결
agent = AgentManager().get_agent()
add_routes(app, agent, path="/upstage_chain")