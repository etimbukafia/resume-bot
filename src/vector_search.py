from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()
from bson import ObjectId

model = SentenceTransformer('all-MiniLM-L6-v2')

uri = os.environ.get("MONGO_URI")
client = MongoClient(uri)
db = client["resumeDB"]
collection = db["resumeCollection"]
vector_index_collection = db["resume_vector_index"]
#ATLAS_VECTOR_SEARCH_INDEX_NAME = "resume_vector_index"

for record in collection.find():
    try:
        resume = record['resume']
        # Directly pass the binary data to pymupdf.open()
        reader = pymupdf.open(stream=resume, filetype="pdf")
        text = ""
            
        # Extract text from all pages
        for page_number in range(reader.page_count):
            page = reader[page_number]
            text += page.get_text() + "\n"

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
        resume_text_chunks = text_splitter.split_text(text)

        # Generate embeddings for the text chunks
        resume_embeddings = [model.encode(chunk) for chunk in resume_text_chunks]

        # Prepare documents for the vector search collection
        documents_for_indexing = [
            {
                "_id": ObjectId(),  # Generate a new unique ID for the vector index document
                "embedding": embedding.tolist(),  # Convert numpy array to list
                "metadata": {
                    "original_id": str(record['_id']),  # Reference to the original document
                    "firstName": record.get('firstName', 'N/A'),
                    "lastName": record.get('lastName', 'N/A'),
                    "created_at": record.get('created_at', None)
                }
            }
            for embedding in resume_embeddings
        ]

        # Insert documents into the vector index collection
        vector_index_collection.insert_many(documents_for_indexing)

        print("vector index populated")
    except Exception as e:
        print(f'An error occurred while processing document ID {record.get("_id")}: {e}')