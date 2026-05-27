# main.py

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from neo4j import GraphDatabase
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
from t5_rewriter import T5Rewriter
import os

# === Load environment variables ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# === Initialize embedding and LLM ===
embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)

# === Connect to ChromaDB ===
chroma_db = Chroma(
    persist_directory="chroma_db",
    collection_name="askneo_qa",
    embedding_function=embedding_model
)

# === Connect to Neo4j ===
driver = GraphDatabase.driver(
    uri="bolt://localhost:7687",
    auth=("neo4j", "neo4j123")
)

# === Initialize T5 Rewriter ===
rewriter = T5Rewriter()

# === Query Neo4j ===
def query_neo4j(disease):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (d:Disease {name: $disease})-[:HAS_PRECAUTION]->(p:Prevention)
            RETURN p.name AS precaution
            """,
            disease=disease
        )
        return [record["precaution"] for record in result]

# === Query ChromaDB with T5 rewritten input ===
def query_chroma(user_input):
    rewritten = rewriter.rewrite(user_input)
    results = chroma_db.similarity_search(rewritten, k=1)
    return results[0].page_content if results else ""

# === Build Prompt for LangChain LLM ===
def build_prompt(user_input, semantic_data, graph_data):
    return [
        SystemMessage(content="You are a helpful health assistant."),
        HumanMessage(content=f"User question: {user_input}"),
        HumanMessage(content=f"Semantic context: {semantic_data}"),
        HumanMessage(content=f"Graph facts (precautions): {graph_data}")
    ]
