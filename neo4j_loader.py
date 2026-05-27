import json
from neo4j import GraphDatabase

# === Neo4j Connection ===
uri = "bolt://localhost:7687"
user = "neo4j"
password = "neo4j123"
driver = GraphDatabase.driver(uri, auth=(user, password))

# === Load JSON Data ===
with open("data/cleaned_qa/qa_pairs.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# === Utility to safely parse comma-separated strings ===
def safe_list(text):
    if isinstance(text, str):
        return [item.strip() for item in text.split(",") if item.strip()]
    return []

# === Clear Existing Graph ===
def clear_graph(tx):
    tx.run("MATCH (n) DETACH DELETE n")

# === Create Graph Structure ===
def create_qa_graph(tx, disease, symptoms, causes, prevention, treatment):
    tx.run("""
        MERGE (d:Disease {name: $disease})

        WITH d
        UNWIND $symptoms AS s
            MERGE (sym:Symptom {name: s})
            MERGE (d)-[:HAS_SYMPTOM]->(sym)

        WITH d
        UNWIND $causes AS c
            MERGE (cause:Cause {name: c})
            MERGE (d)-[:HAS_CAUSE]->(cause)

        WITH d
        UNWIND $prevention AS p
            MERGE (prev:Prevention {name: p})
            MERGE (d)-[:HAS_PREVENTION]->(prev)

        WITH d
        UNWIND $treatment AS t
            MERGE (treat:Treatment {name: t})
            MERGE (d)-[:HAS_TREATMENT]->(treat)
    """, disease=disease,
         symptoms=safe_list(symptoms),
         causes=safe_list(causes),
         prevention=safe_list(prevention),
         treatment=safe_list(treatment)
    )

# === Execute Load ===
with driver.session() as session:
    print("🔄 Clearing old Neo4j graph...")
    session.execute_write(clear_graph)

    print("🚀 Loading new disease data...")
    for entry in qa_data:
        disease_name = entry["disease"].title()
        session.execute_write(
            create_qa_graph,
            disease_name,
            entry["symptoms"],
            entry["causes"],
            entry["prevention"],
            entry["treatment"]
        )
        print(f"✅ Loaded: {disease_name}")

print(f"\n✅ Graph loaded successfully. Total diseases: {len(qa_data)}")
driver.close()
