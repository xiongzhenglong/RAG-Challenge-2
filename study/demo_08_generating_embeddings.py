# study/demo_08_generating_embeddings.py

import json
import os
from pathlib import Path
import sys

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import VectorDBIngestor # VectorDBIngestor handles embedding generation

def main():
    """
    Demonstrates generating text embeddings for selected chunks from a processed report.
    This is a core step in preparing data for Retrieval Augmented Generation (RAG).
    """
    print("Starting text embedding generation demo...")

    # --- 1. Define Paths ---
    # Input is the chunked report (output of demo_07)
    input_chunked_report_dir = Path("study/chunked_reports_output/")
    input_chunked_filename = "report_for_serialization.json" # Assuming this name
    input_chunked_full_path = input_chunked_report_dir / input_chunked_filename

    print(f"Input chunked report directory: {input_chunked_report_dir}")
    print(f"Expected chunked JSON file: {input_chunked_full_path}")

    # --- 2. Prepare Input Data (Load Chunked JSON) ---
    if not input_chunked_full_path.exists():
        print(f"Error: Input chunked JSON file not found at {input_chunked_full_path}")
        print("Please ensure 'demo_07_text_splitting.py' has run successfully.")
        return

    sample_chunks_data = []
    try:
        with open(input_chunked_full_path, 'r', encoding='utf-8') as f:
            chunked_data = json.load(f)
        
        if 'content' in chunked_data and 'chunks' in chunked_data['content'] and chunked_data['content']['chunks']:
            # Extract the first 2-3 chunks for demonstration
            sample_chunks_data = chunked_data['content']['chunks'][:3]
            if not sample_chunks_data:
                 print("No chunks found in the loaded JSON file.")
                 return
            print(f"Successfully loaded chunked JSON. Using {len(sample_chunks_data)} sample chunks for demo.")
        else:
            print("Error: 'content' or 'chunks' not found in the loaded JSON structure.")
            print("Please ensure the input file is correctly formatted (output of demo_07).")
            return
            
    except json.JSONDecodeError:
        print(f"Error: Could not decode the JSON file at {input_chunked_full_path}.")
        return
    except Exception as e:
        print(f"An error occurred while loading the JSON file: {e}")
        return

    # --- 3. Understanding Text Embeddings ---
    # Text embeddings are numerical representations (vectors) of text that capture
    # its semantic meaning. Words, phrases, or entire text chunks with similar meanings
    # will have embeddings that are close together in the vector space.
    #
    # Importance in RAG:
    #   - Semantic Search: When a user asks a question (query), the query is also
    #     converted into an embedding. The RAG system then compares this query
    #     embedding with the embeddings of all stored text chunks.
    #   - Similarity Matching: Chunks whose embeddings are most similar (e.g., by
    #     cosine similarity or dot product) to the query embedding are considered
    #     the most relevant. These relevant chunks are then provided to an LLM
    #     along with the original query to generate an informed answer.
    #
    # API Requirement:
    #   - Generating high-quality embeddings typically requires using pre-trained models
    #     provided by services like OpenAI (e.g., 'text-embedding-ada-002'),
    #     Cohere, Google, etc.
    #   - This usually involves making API calls, which means an API key for the chosen
    #     service must be configured in your environment. For OpenAI, this is the
    #     `OPENAI_API_KEY`. Refer to `study/demo_01_project_setup.py` for how to set this up.

    # --- 4. Generate Embeddings for Sample Chunks ---
    print("\nInitializing VectorDBIngestor to generate embeddings...")
    
    # Check for API key before attempting to initialize or make calls
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Embeddings generation requires an OpenAI API key.")
        print("Please set it up (see demo_01_project_setup.py) and try again.")
        return

    try:
        # VectorDBIngestor sets up the LLM client (e.g., OpenAI) based on environment variables
        # or configuration. It provides methods for embedding generation.
        ingestor = VectorDBIngestor() 
        print("VectorDBIngestor initialized successfully.")
    except Exception as e:
        print(f"Error initializing VectorDBIngestor: {e}")
        print("This might be due to missing API keys or other configuration issues.")
        return

    print("\n--- Generating Embeddings for Sample Chunks ---")
    print("(This involves API calls to an embedding model, e.g., OpenAI's ada-002)")

    for i, chunk_data in enumerate(sample_chunks_data):
        chunk_text = chunk_data.get('text', '')
        chunk_id = chunk_data.get('id', f'sample_chunk_{i+1}')

        if not chunk_text:
            print(f"\nSkipping chunk {chunk_id} as it has no text content.")
            continue

        print(f"\n--- Chunk ID: {chunk_id} ---")
        print(f"  Text (Snippet): \"{chunk_text[:150]}...\"")

        try:
            # `_get_embeddings` expects a list of texts and returns a list of embeddings.
            # For a single chunk, we pass a list containing its text.
            embedding_list = ingestor._get_embeddings([chunk_text])
            
            if embedding_list and isinstance(embedding_list, list) and len(embedding_list) > 0:
                embedding_vector = embedding_list[0] # Get the first (and only) embedding
                
                print(f"  Embedding Generated Successfully.")
                # Most OpenAI embeddings (like ada-002) have 1536 dimensions.
                print(f"  Total Dimensionality: {len(embedding_vector)}")
                # Print the first few dimensions to give an idea of the vector
                print(f"  First 10 Dimensions (Sample): {embedding_vector[:10]}")
            else:
                print("  Error: _get_embeddings did not return the expected list of embeddings.")

        except Exception as e:
            print(f"  Error generating embedding for chunk {chunk_id}: {e}")
            print("  This could be due to API issues (rate limits, key problems), network problems,")
            print("  or issues with the input text format/length for the embedding model.")
            # Optionally, break or continue based on error handling strategy
            # For this demo, we'll continue to try other chunks.
    print("---------------------------------------------")

    print("\nEmbedding generation demo complete.")
    print("Note: Actual storage of these embeddings into a vector database is covered")
    print("in the next steps of a full RAG pipeline (e.g., using VectorDBIngestor.ingest_reports).")

if __name__ == "__main__":
    main()
