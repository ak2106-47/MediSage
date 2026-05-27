# qa_chain.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Set up the LLM
llm = ChatOpenAI(
    openai_api_key=openai_key,
    temperature=0.7,
    max_tokens=1024,
    model="gpt-3.5-turbo"
)

# Define prompt template
hybrid_prompt_template = PromptTemplate(
    input_variables=["question", "semantic", "graph"],
    template="""
You are a highly knowledgeable medical assistant. Use both semantic search results and graph-based facts to answer clearly and completely.

User Question: {question}

Relevant Medical Info:
- Semantic Context: {semantic}
- Structured Graph Insights: {graph}

Answer:
"""
)

# Wrap into LLMChain
hybrid_chain = LLMChain(
    llm=llm,
    prompt=hybrid_prompt_template
)
