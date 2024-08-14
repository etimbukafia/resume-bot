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
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from utils import fireworks_llm

llm = fireworks_llm
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
    
    def embed_query(self, texts: str) -> float:
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

# Initialize your custom embedding class
model_name = 'all-MiniLM-L6-v2'
embeddings = SentenceTransformerEmbeddings(model_name)"""

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("loaded embeddings model")

# Instantiate vector store
vector_store = MongoDBAtlasVectorSearch(
   collection=vector_index_collection,
   embedding=embeddings,
   index_name="vector_index",
   embedding_key="embedding",
   text_key = "summary"
)
print("Instantiated vector store")

retriever = vector_store.as_retriever()

# Define prompt template
template = """
Use the following pieces of context to answer the question at the end.
{context}
Question: {question}
"""
custom_rag_prompt = PromptTemplate.from_template(template)

def format_docs(docs):
   return "\n\n".join(doc.page_content for doc in docs)

# Create chain
rag_chain = (
   {"context": retriever | format_docs, "question": RunnablePassthrough()}
   | custom_rag_prompt
   | llm
   | StrOutputParser()
)

query = "Which of the applicants worked as a Data Entry Clerk"

answer = rag_chain.invoke(query)

print(answer)