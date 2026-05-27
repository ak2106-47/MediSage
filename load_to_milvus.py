import os
import json
from dotenv import load_dotenv
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from sentence_transformers import SentenceTransformer

# === Load environment variables ===
load_dotenv()
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")  # Use IP for Docker/Windows
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# === Connect to Milvus ===
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
print(f"✅ Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

# === Collection setup ===
collection_name = "askneo_qa"
if utility.has_collection(collection_name):
    print(f"⚠️ Collection '{collection_name}' exists. Dropping it...")
    Collection(collection_name).drop()

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="disease", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="qa_type", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
]
schema = CollectionSchema(fields, description="Medical Q&A pairs")
collection = Collection(name=collection_name, schema=schema)
print(f"✅ Created collection: {collection_name}")

# === Load model and data ===
model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/cleaned_qa/qa_pairs.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# === Prepare vectors and metadata ===
embeddings, diseases, qa_types, texts = [], [], [], []

for item in qa_data:
    disease = item["disease"]
    for qa_type in ["symptoms", "causes", "prevention", "treatment"]:
        text = item.get(qa_type, "").strip()
        if not text:
            continue
        emb = model.encode(text)
        embeddings.append(emb)
        diseases.append(disease)
        qa_types.append(qa_type)
        texts.append(text)

# === Insert into Milvus ===
print(f"📦 Inserting {len(embeddings)} QA entries...")
collection.insert([embeddings, diseases, qa_types, texts])
collection.flush()
print("✅ Successfully inserted all Q&A into Milvus.")
