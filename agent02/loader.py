from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader

# PDF loader from langchain_community PyPDFLoader
def load_pdf(path: str)->List[Document]:
    """
    Loads a PDF file using PyPDFLoader.
    
    Args:
        path (str): The file path to the PDF.
        
    Returns:
        List[Document]: A list of Documents, one for each page.
    """
    # loading pdf file

    print(f"--- Loading PDF with PyPDFLoader: {path} ---")
    loader = PyPDFLoader(path)
    return loader.load()

# Txt loader from langchain_community TextLoader
def load_txt(path: str)->List[Document]:
    """
    Loads a txt file using TextLoader

    Args: 
        path (str): The file path to the TXT
    
    Returns:
        List[Document]: A list of Documents
    """

    print(f"--- Loading TXT with TextLoader: {path} ---")
    loader = TextLoader(path)
    return loader.load()

def load_web(url: str)->List[Document]:
    """
    Loads a HTML text from webpage using WebBaseLoader
    Reference: https://docs.langchain.com/oss/python/integrations/document_loaders/web_base
    Args:
        url (str): The URL to the web page.
    
    Returns:
        List[Document]: A list of Documents
    """
    print(f"--- Loading Web with WebBaseLoader: {url} ---")
    loader = WebBaseLoader(url)
    return loader.load()

# Document loader interface

class DocumentLoader:  # Combined Document loader from mutliple types

    @staticmethod
    def load(source: str)-> List[Document]:
        """
        Intelligently loads a document based on the source string.
        
        - URLs (http/https) -> WebBaseLoader
        - .pdf files -> PyPDFLoader
        - Other files -> TextLoader
        
        Args:
            source (str): The file path or URL.
            
        Returns:
            List[Document]: The loaded documents.
        """
        if source.startswith("http://") or source.startswith("https://"):
            return load_web(source)

        elif source.endswith(".pdf"):
            return load_pdf(source)
        else:
            return load_txt(source)

__all__ = ["load_pdf", "load_text", "load_web", "DocumentLoader"]