from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_mongodb import MongoDBAtlasVectorSearch
from sentence_transformers import SentenceTransformer
import pprint
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List

uri = os.environ.get("MONGO_URI")
client = MongoClient(uri)
db = client["resumeDB"]
vector_index_collection = db["resume_vector_index"]
print("connected to database")

"""class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self.model.encode(documents)
    
    def embed_query(self, texts: str):
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

# Initialize your custom embedding class
model_name = 'all-MiniLM-L6-v2'
embedding = SentenceTransformerEmbeddings(model_name)"""

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("loaded embeddings model")

# Instantiate vector store
vector_store = MongoDBAtlasVectorSearch(
   collection=vector_index_collection,
   embedding=embeddings,
   index_name="vector_index",
   embedding_key="embedding"
)
print("Instantiated vector store")


query = "Alexa Text-to-Speech Research team"
print("embedding query")
#query_result = embedding.embed_query(query)
print("performing similarity search")
results = vector_store.similarity_search(query)
pprint.pprint(results)

