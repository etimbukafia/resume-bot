from dotenv import load_dotenv
load_dotenv()
#from sentence_transformers import SentenceTransformer
from typing import List
from db_connect import Database
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from models import QueryRequest
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from chain import retrieval_chain
from config import Configs

configs = Configs()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = None
llm = None


@app.on_event("startup")
async def startup_event():
    global vector_store, llm

    # Initialize the database connection
    await configs.initialize()

    vector_store = configs.get_vector_store()
    print("Vector store initialized")

    #Load your LLM
    llm = configs.get_llm()
    print("LLM loaded")

@app.on_event("shutdown")
async def shutdown_db_client():
    await Database.close()


@app.get('/chat')
async def resume_chat(query: QueryRequest):
    try:

        answer = await retrieval_chain(vector_store, llm, query.query)
        if answer is None:
            raise HTTPException(status_code=500, detail="Failed to retrieve answer")
        return {answer}
    
    except Exception as e:
        print(f"Error in resume_chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
