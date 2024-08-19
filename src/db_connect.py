from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import MongoDBAtlasVectorSearch

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    _client = None
    _db = None
    _resume_collection = None
    _vector_collection = None
    _vector_store = None

    @classmethod
    async def connect(cls, embeddings):
        if cls._client is None:
            try:
                uri = os.environ.get("MONGO_URI")
                if uri is None:
                    raise ValueError("MONGO_URI environment variable is not set.")
                print(f"Connecting to MongoDB")

                db_name = "resumeDB"
                resume_collection_name = "resumeCollection"
                vector_collection_name = "resume_vector_index"

                cls._client = AsyncIOMotorClient(uri, maxPoolSize=50)
                cls._db = cls._client[db_name]
                cls._resume_collection = cls._db[resume_collection_name]
                cls._vector_collection = cls._db[vector_collection_name]

                cls._vector_store = MongoDBAtlasVectorSearch.from_connection_string(
                    uri,
                    db_name + "." + vector_collection_name,
                    embedding=embeddings,
                    index_name="vector_index",
                    embedding_key="embedding",
                    text_key="summary"
                )

                logger.info("Connected to MongoDB and initialized vector store instance")
            except ConnectionError as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
        return cls._resume_collection, cls._vector_collection, cls._vector_store
    
    @classmethod
    async def close(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            cls._resume_collection = None
            cls._vector_collection = None
            cls._vector_store = None
            logger.info("Disconnected from MongoDB")