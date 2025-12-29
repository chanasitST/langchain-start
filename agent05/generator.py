from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agent05.models import Quiz

def generate_quiz(topic: str, retrievers: BaseRetriever = None) -> Quiz:
    """_summary_
    Generates a quiz for a given topic using RAG (optional) and structured output.

    Args:
        topic (str): _description_
        retrievers (BaseRetriever, optional): _description_. Defaults to None.

    Returns:
        Quiz: _description_
    """
    print(f"Generating quiz for topic: {topic}")

    context = ""
    if retrievers:
        # Step 1 Retrieve relevant chunks from the retriever
        print("--- Retrieving relevant context ---")
        docs = retriever.invoke(topic)

        # Step 2 Combine retrieved chunks into a context string
        context = "\n".join([doc.page_content for doc in docs])
        # context = "\n\n".join([doc.page_content for doc in docs])
    else:
        print("--- No retriever provided, skipping context retrieval (using existing LLM knowledge) ---")
    
    # Step 3 Create prompt template
    template = """You are an expert teacher creating a quiz to test student understanding.

    Context:
    {context}

    Topic: {topic}

    Instructions:
    - Create a quiz with 3 multiple-choice questions based on the context.
    - Each question should have 4 options.
    - Ensure the questions test understanding, not just memorization.
    - Provide a clear explanation for the correct answer.
    - The output must be a valid JSON object matching the Quiz schema.
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    # Step 4 Init LLM with structured output
    llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
    structured_llm = llm.with_structured_output(Quiz)
    # https://docs.langchain.com/oss/python/langchain/structured-output