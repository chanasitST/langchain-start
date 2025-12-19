from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def generate_explanation(question: str, retriever: BaseRetriever) -> str:
    """
    Generates a clear, simple explanation for a give question using RAG.

    Args:
        question (str): The user question
        retriever (BaseRetriever) : The retriever to find the relevant context from chunk

    Returns:
        str: The generated explaination

    """
    
    print(f"\n ? Question: {question}")

    # Step 1 Retrieve
    print("--- Base Retriever is finding the relevant context")
    
    docs = retriever.invoke(question)
    
    print(f"Retrieved {len(docs)} chunks")

    # Step 2 Combine context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Step 3 Create prompt
    # This prompt is optimized for simple, clear explanations
    template = """
    You are a helpful assistant that provides clear and concise explanations for a given question.

    Context:
    {context}

    Question: {question}

    Instructions:
    - Use the context above to answer the question
    - Explain it as if you're talking to someone learning this for the first time
    - Use simple language and avoid jargon when possible
    - If you use technical terms, explain them briefly
    - Be concise but complete
    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)
    # define reusable prompt templates
    # https://reference.langchain.com/python/langchain_core/prompts/
    """
    # Example usage system prompt + user prompt
    from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
    chatprompt2 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant with a {personality} personality"),
    HumanMessagePromptTemplate.from_template("{input}")
    ])
    """

    # Step 4 init llm
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    # Step 5 Create prompt chain
    chain = prompt | llm

    # Chains -> link multple steps together eg. prompt + llm

    # Step 6 generate explanation
    print("--- Generating Explaination invoking prompt chain ---")
    response = chain.invoke({"context": context, "question": question})

    return response.content
