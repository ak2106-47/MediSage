from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Load env
load_dotenv()
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# Connect to Milvus
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample query
query_text = "What are the common symptoms of diabetes?"
query_vector = model.encode(query_text).tolist()

# Load collection
collection = Collection("disease_qa")
collection.load()

# Run vector search
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=3,
    output_fields=["disease", "qa_type", "text"]
)

# Display results
print(f"\n🔎 Top matches for: \"{query_text}\"\n")
for hit in results[0]:
    print(f"✅ [{hit.entity.get('qa_type')}] {hit.entity.get('disease')}")
    print(f"   → {hit.entity.get('text')}\n")
