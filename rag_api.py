from fastapi import FastAPI
from langserve import add_routes

from langchain_api_web import chain

app = FastAPI()

@app.get("/")
async def root():
    return {"message":"Hello world"}

add_routes(app, chain, path="/upstage_chain")

