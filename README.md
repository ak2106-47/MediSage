# MediSage 🩺

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

MediSage is a conversational health-information assistant that answers questions about diseases, symptoms, causes, treatments, and prevention. Rather than relying on a single lookup method, it blends two complementary sources of knowledge — a vector store for meaning-based search and a graph database for structured medical relationships — and lets a language model weave their results into a single, readable answer. A lightweight T5 model rephrases each question first, so the retrieval step has a cleaner query to work with.

> ⚠️ **Medical disclaimer:** MediSage is an educational and informational tool only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider about any medical concern.

---

## Why two databases?

Most question-answering tools pick one retrieval strategy and live with its blind spots. MediSage runs both in parallel:

- **Semantic search (ChromaDB).** Embeds the question as a vector and finds the closest matching medical Q&A pairs — great for fuzzy, natural-language phrasing where exact keywords don't line up.
- **Graph traversal (Neo4j).** Looks up the disease entity in a knowledge graph and pulls its connected symptoms, causes, and treatments — great for precise, relationship-driven facts.

The two result sets are then fused and handed to the LLM, which produces a final answer grounded in both. This hybrid design tends to be more reliable than either approach on its own.

---

## The answer pipeline

A question flows through MediSage in five stages:

1. **Rephrase** — A T5 model paraphrases the raw question into a cleaner form for retrieval.
2. **Analyze** — The rewritten query is scanned to identify the disease entity and the user's intent.
3. **Retrieve (semantic)** — ChromaDB returns the most similar Q&A pairs by vector distance.
4. **Retrieve (graph)** — Neo4j returns structured relationships for the matched disease via a Cypher query.
5. **Synthesize** — LangChain merges both result sets into a prompt, and GPT-3.5-turbo writes the final response with supporting context.

```
Question
   │
   ▼  T5 paraphrase  →  intent + entity detection
   │
   ├──▶ ChromaDB  (vector similarity)
   └──▶ Neo4j     (Cypher graph query)
   │
   ▼  fuse results → LangChain prompt → GPT-3.5-turbo
   │
   ▼
Answer (+ reasoning context)
```

---

## What's included

- Hybrid retrieval that combines vector search and graph queries on every question.
- T5-based query rewriting to improve retrieval quality before search runs.
- A Streamlit web app with both typed and voice input.
- A command-line mode for quick terminal use.
- An optional Milvus vector-database backend, set up via Docker Compose.
- Cached embeddings and parallel retrieval to keep responses fast.
- A medical dataset spanning diseases, symptoms, causes, treatments, and prevention.

---

## Tools under the hood

| Role | Technology |
|------|------------|
| Web UI | Streamlit |
| Query rewriting | T5-small |
| Answer generation | OpenAI GPT-3.5-turbo |
| Embeddings | OpenAI `text-embedding-ada-002` (Sentence Transformers as a fallback) |
| Vector store | ChromaDB (Milvus optional) |
| Knowledge graph | Neo4j |
| Orchestration | LangChain |

---

## Getting it running

### You'll need

- Python 3.8 or newer
- An OpenAI API key
- A Neo4j database (local install or Neo4j AuraDB in the cloud)
- Docker — only if you want to use the optional Milvus backend

### Setup

Clone the project and install its dependencies:

```bash
git clone https://github.com/<your-username>/medisage.git
cd medisage
pip install -r requirements.txt
```

Create your environment file and drop in your credentials:

```bash
cp .env.example .env
```

```bash
# .env
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j123
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

Make sure Neo4j is up (the defaults expect `bolt://localhost:7687`), then load the medical data into both stores:

```bash
python neo4j_loader.py     # populate the knowledge graph
python load_to_chroma.py   # populate the vector store
```

Finally, launch it however you prefer:

```bash
streamlit run app.py   # web interface at http://localhost:8501
# — or —
python main.py         # command-line interface
```

---

## Using MediSage

**In the browser:** open `http://localhost:8501`, then type or speak a question such as *"What are the symptoms of diabetes?"* The answer appears along with the context used to produce it.

**In the terminal:** run `python main.py` and enter your question at the prompt.

**From your own code:** the core functions can be imported directly.

```python
from main import query_chroma, query_neo4j, build_prompt, llm

semantic = query_chroma("diabetes symptoms")   # vector search
graph    = query_neo4j("diabetes")             # graph lookup

prompt = build_prompt("diabetes symptoms", semantic, graph)
answer = llm(prompt)
```

---

## Layout of the repo

```
medisage/
├── app.py                 # Streamlit web app
├── main.py                # Command-line entry point
├── hybrid_qa_chain.py     # Combined vector + graph retrieval
├── qa_chain.py            # LangChain wiring
├── t5_rewriter.py         # T5 query paraphrasing
├── neo4j_loader.py        # Loads data into Neo4j
├── load_to_chroma.py      # Loads data into ChromaDB
├── embed_and_store.py     # Embedding + storage helpers
├── docker-compose.yaml    # Optional Milvus container
├── data/
│   ├── cleaned_qa/
│   │   └── qa_pairs.json
│   └── disease_list.txt
├── chroma_db/             # Local ChromaDB store
├── assets/                # UI assets
└── logs.txt               # Runtime logs
```

---

## Configuration notes

- **Neo4j** holds the medical knowledge graph — the disease → symptom / cause / treatment relationships that drive structured lookups.
- **ChromaDB** is the default vector store for semantic similarity.
- **Milvus** is an optional, higher-throughput alternative to ChromaDB; a ready-to-go `docker-compose.yaml` is included if you'd like to try it.

---

## If something goes wrong

**Neo4j won't connect** — Confirm the database is actually running and that the URI, username, and password in `.env` match your instance.

**OpenAI calls fail** — Double-check the API key, and verify your account still has quota and active billing.

**ChromaDB acting up** — Restart the service and make sure nothing else is holding port `8000`.

---

## Where the data comes from

The bundled medical knowledge is assembled from disease-to-symptom mappings, treatment and prevention notes, cause-and-effect links, and curated medical Q&A pairs — all intended for learning and experimentation rather than clinical use.

---

## Ideas for later

- Support for additional languages
- Hooks into external medical APIs
- Predictive modeling for likely conditions
- A companion mobile app
- Connections to wearable and health-monitoring data

---

## Contributing

Contributions are welcome. The usual flow works well: fork the repo, branch off for your change, commit your work (with tests where it makes sense), and open a pull request.

---

## License

Released under the MIT License. See the [LICENSE](LICENSE) file for the full text.
