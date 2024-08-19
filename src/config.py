from db_connect import Database
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_fireworks import ChatFireworks
import asyncio

load_dotenv()

class Configs:
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None

    async def initialize(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _, _, self.vector_store = await Database.connect(self.embeddings)
        self.llm = ChatFireworks(
            model="accounts/fireworks/models/mixtral-8x7b-instruct",
            temperature=0.7,
            max_tokens=2048,
            model_kwargs={"top_p": 1},
            cache=None,
        )

    def get_llm(self):
        return self.llm

    def get_embeddings(self):
        return self.embeddings

    def get_vector_store(self):
        return self.vector_store

