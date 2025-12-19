import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

def chunk_documents(docs: List[Document])-> List[Document]:

    print("--- Chunking Documents (Semantic) ---")
    # initialize embeddings using openai api key
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # initialize semantic chunker
    # breakpoint_threshold_type="percentile" is a good default
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )

    """
- `percentile` (default) — In this method, all differences between sentences are calculated, and then any difference greater than the X percentile is split.

- `standard_deviation` — In this method, any difference greater than X standard deviations is split.

- `interquartile` — In this method, the interquartile distance is used to split chunks.
    """
    
    # split documents
    chunks = text_splitter.split_documents(docs)
    
    print(f"--- Generated {len(chunks)} chunks from {len(docs)} documents ---")
    return chunks