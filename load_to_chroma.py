import json
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load data
with open("data/cleaned_qa/qa_pairs.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Chroma setup
chroma_path = "chroma_db"
chroma_client = chromadb.Client(Settings(persist_directory=chroma_path, anonymized_telemetry=False))
chroma_client.delete_collection("askneo_qa")  # reset for clean load
collection = chroma_client.create_collection(name="askneo_qa")

# Embeddings
embedding_model = OpenAIEmbeddings()

# Prepare docs
texts = [item["question"] for item in qa_data]
metadatas = [{"answer": item["answer"]} for item in qa_data]

# Embed & add
db = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    metadatas=metadatas,
    persist_directory=chroma_path,
    collection_name="askneo_qa"
)
db.persist()
print("✅ Loaded into ChromaDB.")
