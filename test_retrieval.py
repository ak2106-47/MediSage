# test_retrieval.py

from chromadb import HttpClient
from sentence_transformers import SentenceTransformer

# Step 1: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Connect to persistent ChromaDB server
client = HttpClient(host="localhost", port=8000)
collection = client.get_collection("askneo_medical")

# Step 3: Example query
query = "What are the symptoms of asthma?"
print(f"\n🔍 Query: {query}")

# Step 4: Embed query and search
query_embedding = model.encode([query])

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,
    include=["documents", "metadatas"]
)

# Step 5: Print results
print("\n📄 Top Matches:")
for i, doc in enumerate(results["documents"][0]):
    print(f"\nResult #{i+1}")
    print("Question:", doc)
    print("Answer:", results["metadatas"][0][i]["answer"])
