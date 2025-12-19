from typing import List
from langchain_core import embeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# https://reference.langchain.com/python/langchain_core/retrievers/?h=baseretriever
"""
A retrieval system is defined as something that can take string queries and return the most 'relevant' documents from some source.
"""

class ManualEnsembleRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    weights: List[float]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        scored_docs = {}

        for retriever, weight in zip(self.retrievers, self.weights):
            # docs = retriever.get_relevant_documents(query)
            docs = retriever.invoke(query)
            for i, doc in enumerate(docs):
                key = doc.page_content
                scored_docs.setdefault(key, [doc, 0])
                scored_docs[key][1] += weight * (1 / (i + 1))

        return [
            v[0] for v in
            sorted(scored_docs.values(), key=lambda x: x[1], reverse=True)
        ]

def create_hybrid_retriever(docs: List[Document]) -> BaseRetriever:
    """
    Creates a Hybrid Retriever BM25 + Vector Search

    """

    print("--- Creating Hybrid Retriever ---")
    """
         BM25 also known as the Okapi BM25, is a ranking function used in information retrieval systems to estimate the relevance of documents to a given search query. BM25Retriever retriever uses the rank_bm25 package. 
    """

    # Step 1 BM25 Retriever (sparse / keyword)
    # BM25 for exact matches and specific terms
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5
    # will return top 5 documents reranked by BM25 algorithm

    # Step 2 Vector Retriever (dense / semantic)
    # Semantic matcheds, good for conceptual matching
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="documents",  # tutorial name = day_03_hybrid
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    # https://reference.langchain.com/python/langchain_core/vectorstores/#langchain_core.vectorstores.in_memory.InMemoryVectorStore.as_retriever
    # search_kwargs: Keyword arguments to pass to the search function.

    # Step 3 Combine Retrievers
    # Ensemble / Combine both with equal weight (.5 / .5)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.5, 0.5]
        )
    
    # ensemble_retriever = ManualEnsembleRetriever(
    # retrievers=[bm25_retriever, vector_retriever],
    # weights=[0.5, 0.5]
    # )

    return ensemble_retriever