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