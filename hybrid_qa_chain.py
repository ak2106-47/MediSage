# hybrid_qa_chain.py

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb import HttpClient
from neo4j import GraphDatabase

# === Load Environment ===
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# === Initialize LLM & Embeddings ===
llm = ChatOpenAI(openai_api_key=openai_key, temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === Connect to ChromaDB ===
chroma_client = HttpClient(host="localhost", port=8000)
chroma_collection = chroma_client.get_collection("askneo_medical")

# === Connect to Neo4j ===
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_pass = "neo4j123"
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))

# === Semantic Engine Retrieval ===
def get_semantic_answer(query):
    query_embedding = embedding_model.embed_query(query)
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas"]
    )
    docs = results["metadatas"][0]
    combined = "\n\n".join([f"Q: {doc['original_question']}\nA: {doc['answer']}" for doc in docs])
    return combined

# === Graph Engine Retrieval ===
def get_graph_answer(query):
    # Very simple keyword-based Cypher template (can improve later)
    if "symptom" in query.lower():
        relation = "HAS_SYMPTOM"
        label = "Symptom"
    elif "cause" in query.lower():
        relation = "HAS_CAUSE"
        label = "Cause"
    elif "treatment" in query.lower():
        relation = "HAS_TREATMENT"
        label = "Treatment"
    elif "prevent" in query.lower():
        relation = "HAS_PREVENTION"
        label = "Prevention"
    else:
        return "⚠️ No matching graph query could be formed."

    # Extract disease keyword (simplified)
    disease = query.split()[-1].capitalize()

    cypher = f"""
    MATCH (d:Disease)-[:{relation}]->(x:{label})
    WHERE toLower(d.name) CONTAINS toLower("{disease}")
    RETURN d.name AS disease, collect(x.name) AS items
    """

    with neo4j_driver.session() as session:
        result = session.run(cypher)
        records = result.single()
        if not records:
            return "⚠️ No graph data found for that query."
        items = records["items"]
        return f"{label}s related to {records['disease']}: {', '.join(items)}"

# === Final Answer Synthesizer ===
def synthesize_answer(user_query, semantic_result, graph_result):
    prompt = f"""
You are a medical assistant AI named MediSage.

A user asked: "{user_query}"

You searched two sources:
1. Semantic Knowledge Base (ChromaDB):
{semantic_result}

2. Structured Medical Graph (Neo4j):
{graph_result}

Now synthesize the best possible answer, combining both sources with clarity and accuracy.
"""
    return llm.invoke(prompt).content

# === Run Demo ===
if __name__ == "__main__":
    user_query = input("🔍 Enter your medical question: ")

    print("\n⏳ Running semantic engine...")
    semantic_result = get_semantic_answer(user_query)
    print("✅ Got semantic result:", semantic_result)

    print("\n⏳ Running graph engine...")
    graph_result = get_graph_answer(user_query)
    print("✅ Got graph result:", graph_result)

    print("\n🤖 Calling LLM to synthesize final answer...")
    final_answer = synthesize_answer(user_query, semantic_result, graph_result)
    print("✅ Final Answer:\n", final_answer)
