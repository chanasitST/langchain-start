import os
from loader import DocumentLoader

def read_text_from_path(path: str):
    with open(path, "r") as f:
        content = f.read()
    return content
    
def create_dummy_text_file(path: str):
    """Creates a dummy text file for testing."""
    with open(path, "w") as f:
        f.write("This is a sample text file for Day 2 of LangChain 2026.\n")
        f.write("It contains multiple lines to test loading.\n")
        f.write("LangChain is awesome!")
    print(f"--- Created dummy file: {path} ---")

def cleanup_file(path: str):
    """Removes a file if it exists."""
    if os.path.exists(path):
        os.remove(path)
        print(f"--- Cleaned up file: {path} ---")

def main():
    # Step 1 Read text from path 
    text = read_text_from_path("data.txt")
    # print(f"Text content: {text}")

    # Step 2 Draft multiple sources (path)
    # sources = [text, "https://www.notion.so/Context-Engineering-for-Agents-2a1808527b17803ba221c2ced7eef508"]
    sources = ["data.txt", "https://www.notion.so/Context-Engineering-for-Agents-2a1808527b17803ba221c2ced7eef508"]
    

    # Step 3 Load the sources

    for source in sources:
        print(f"Loading source: {source}")
        try:
            documents = DocumentLoader.load(source)
            print(f"✅ Successfully loaded {len(documents)} document(s).")
            
            if documents:
                first_doc = documents[0]
                content_preview = first_doc.page_content[:150].replace('\n', ' ')  # Preview first 150 chars
                print(f"   📄 Metadata: {first_doc.metadata}")  # display metadata
                print(f"   📝 Content Preview: {content_preview}...")
        
        except Exception as e:
            print(f"Failed to load {source}")
            print(f"Error: {e}")

    # Step 4 Clean up
    # cleanup_file("data.txt")


if __name__ == "__main__":
    main()