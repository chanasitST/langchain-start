import sys

# Check what's available in langchain_community.retrievers
try:
    import langchain_community.retrievers as retrievers
    print("Available in langchain_community.retrievers:")
    print([x for x in dir(retrievers) if not x.startswith('_')])
except Exception as e:
    print(f"Error importing langchain_community.retrievers: {e}")

# Try to find EnsembleRetriever
print("\n--- Searching for EnsembleRetriever ---")

try:
    from langchain.retrievers import EnsembleRetriever
    print("✓ Found: from langchain.retrievers import EnsembleRetriever")
except ImportError as e:
    print(f"✗ langchain.retrievers: {e}")

try:
    from langchain_community.retrievers import EnsembleRetriever
    print("✓ Found: from langchain_community.retrievers import EnsembleRetriever")
except ImportError as e:
    print(f"✗ langchain_community.retrievers: {e}")

try:
    from langchain_core.retrievers import EnsembleRetriever
    print("✓ Found: from langchain_core.retrievers import EnsembleRetriever")
except ImportError as e:
    print(f"✗ langchain_core.retrievers: {e}")