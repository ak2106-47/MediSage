# embed_and_store.py

import json
import uuid
from sentence_transformers import SentenceTransformer
from chromadb import HttpClient

# Step 1: Load paraphrased Q&A
with open("qa_paraphrase.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Step 2: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Connect to running Chroma server
client = HttpClient(host="localhost", port=8000)

# Step 4: Get or create collection
collection = client.get_or_create_collection("askneo_medical")

# Step 5: Prepare data
texts = [item["paraphrased_question"] for item in data]
metadatas = [{"answer": item["answer"], "original_question": item["original_question"]} for item in data]
ids = [str(uuid.uuid4()) for _ in data]

# Step 6: Embed and store
print("🔄 Embedding...")
embeddings = model.encode(texts, show_progress_bar=True)

print("💾 Storing in ChromaDB...")
collection.add(
    embeddings=embeddings,
    documents=texts,
    metadatas=metadatas,
    ids=ids
)

print(f"✅ Stored {len(texts)} entries in persistent ChromaDB.")
