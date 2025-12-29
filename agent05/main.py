import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from agent02.loader import DocumentLoader
from agent03.chunker import chunk_documents
from agent03.retriever import create_hybrid_retriever
from agent05.generator import generate_quiz

load_dotenv()