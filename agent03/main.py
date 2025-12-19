import sys
import os
import shutil
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent02.loader import DocumentLoader
from chunker import chunk_documents
from retriever import create_hybrid_retriever

load_dotenv()

def create_knowledge_base(path: str):
    # create a dummy text file with enough length for chunking

    content = """
    LangChain is a framework for developing applications powered by language models.
    It enables applications that:
    - Are context-aware: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
    - Reason: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)
    
    The main value props of LangChain are:
    1. Components: composable tools and integrations for working with language models. Components are modular and easy-to-use, whether you are using the rest of the LangChain framework or not.
    2. Off-the-shelf chains: built-in assemblages of components for accomplishing higher-level tasks.
    
    LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. 
    Compared to other LLM frameworks, it offers these core benefits: cycles, controllability, and persistence.
    LangGraph allows you to define flows that involve cycles, essential for most agentic architectures.
    
    Semantic Chunking is a technique to split text based on meaning. 
    Instead of splitting by a fixed number of characters, it uses embeddings to find "breakpoints" where the topic changes.
    This results in chunks that are semantically complete, improving retrieval quality.
    """
    
    with open(path, "w") as f:
        f.write(content.strip())
    print(f"--- created knowledge base ({path}) ---")

def cleanup(path: str):
    # removes the files and chroma db
    if os.path.exists(path):
        os.remove(path)
    if os.path.exists("chroma.db"):
        shutil.rmtree("chroma.db")

# --- Execution main loop --- #

def main():
    print("main loop")
    
    # Step 1 create knowledge base txt
    file_path = "knowledge.txt"
    create_knowledge_base(file_path)

    try:
        # Step 2 load the knowledge base
        docs = DocumentLoader.load(file_path)
        print(f"Loaded {len(docs)} raw unstructed documents")

        # Step 3 Chunking the documents
        chunks = chunk_documents(docs)
        print(f"Generated {len(chunks)} semantic chunks")

        # Step 4 Index & Retrieve
        retriever = create_hybrid_retriever(chunks)

        # Step 5 Query
        query = "What is the main value props of LangChain?"
        results = retriever.invoke(query)
        print(f"Retrieved {len(docs)} documents")

        # preview the content
        for i, doc in enumerate(results):
            print(f"\n Result {i+1}")
            content_preview = doc.page_content[:150].replace('\n', '')
            print(f" {content_preview}...")
        
    except Exception as e:
        print(f"Error : {e}")
    finally:
        # Step 6 clean
        print("\n")
        cleanup(file_path)

if __name__ == "__main__":
    main()