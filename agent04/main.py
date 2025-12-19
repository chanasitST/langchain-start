import os
from dotenv import load_dotenv
from agent02.loaders import DocumentLoader
from agent03.chunking.chunker import chunk_documents
from agent03.chunking.retriever import create_hybrid_retriever
from agent04.explainer import generate_explanation


load_dotenv()