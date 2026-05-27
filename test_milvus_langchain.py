from langchain_community.vectorstores import Milvus
from sentence_transformers import SentenceTransformer
from pymilvus import connections
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()
host = os.getenv("MILVUS_HOST", "localhost")
port = os.getenv("MILVUS_PORT", "19530")

# Connect to Milvus
connections.connect("default", host=host, port=port)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load LangChain Milvus vector store
milvus_db = Milvus(
    embedding_function=model,
    collection_name="askneo_qa"
)

# Query test
query = "What are the symptoms of dengue?"
results = milvus_db.similarity_search(query, k=3)

print(f"\n🔍 Top matches for: \"{query}\"\n")
for i, doc in enumerate(results):
    print(f"{i+1}. {doc.page_content}\n")
