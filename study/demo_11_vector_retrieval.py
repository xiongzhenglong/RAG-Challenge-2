# study/demo_11_vector_retrieval.py

import json
import os
from pathlib import Path
import sys
import faiss
import numpy as np
from openai import OpenAI # For generating query embeddings
from dotenv import load_dotenv # For loading OPENAI_API_KEY from .env

# Load environment variables from .env file (especially OPENAI_API_KEY)
load_dotenv()

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """
    Demonstrates retrieving relevant text chunks from a FAISS vector database
    based on a sample query. This involves generating an embedding for the query
    and using FAISS to find the most similar chunk embeddings.
    """
    print("Starting vector retrieval demo...")

    # --- 1. Define Paths ---
    # Input chunked report (output of demo_07)
    input_chunked_report_dir = Path("study/chunked_reports_output/")
    input_chunked_filename = "report_for_serialization.json" # Assuming this name
    input_chunked_report_path = input_chunked_report_dir / input_chunked_filename

    # Input FAISS index (output of demo_09)
    faiss_index_dir = Path("study/vector_dbs/")
    faiss_index_filename = "demo_report.faiss" # Assuming this name from demo_09
    faiss_index_path = faiss_index_dir / faiss_index_filename

    print(f"Input chunked report path: {input_chunked_report_path}")
    print(f"FAISS index path: {faiss_index_path}")

    # --- 2. Prepare Data (Load Chunked JSON and FAISS Index) ---
    if not input_chunked_report_path.exists():
        print(f"Error: Input chunked JSON file not found at {input_chunked_report_path}")
        print("Please ensure 'demo_07_text_splitting.py' has run successfully.")
        return
    if not faiss_index_path.exists():
        print(f"Error: FAISS index file not found at {faiss_index_path}")
        print("Please ensure 'demo_09_creating_vector_db.py' has run successfully.")
        return

    chunks = []
    try:
        with open(input_chunked_report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        if 'content' in report_data and 'chunks' in report_data['content'] and report_data['content']['chunks']:
            chunks = report_data['content']['chunks']
            if not chunks:
                print("No chunks found in the loaded JSON file. Cannot perform retrieval.")
                return
            print(f"Successfully loaded chunked JSON. Found {len(chunks)} chunks.")
        else:
            print("Error: 'content' or 'chunks' not found in the loaded JSON structure.")
            return
    except json.JSONDecodeError:
        print(f"Error: Could not decode the JSON file at {input_chunked_report_path}.")
        return
    except Exception as e:
        print(f"An error occurred while loading the chunked JSON: {e}")
        return

    try:
        faiss_index = faiss.read_index(str(faiss_index_path))
        print(f"Successfully loaded FAISS index. Index contains {faiss_index.ntotal} vectors.")
        if faiss_index.ntotal != len(chunks):
            print(f"Warning: Number of vectors in FAISS index ({faiss_index.ntotal}) "
                  f"does not match number of chunks in JSON ({len(chunks)}). "
                  "This might lead to incorrect retrieval mapping.")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return

    # --- 3. Understanding Vector Retrieval ---
    # Vector retrieval is the process of finding the most relevant items (text chunks)
    # from a collection based on their semantic similarity to a user's query.
    # The process typically involves:
    #   1. Generating an Embedding for the Query: The user's query (natural language)
    #      is converted into a numerical vector (embedding) using the same embedding
    #      model that was used to create embeddings for the text chunks.
    #   2. Searching the Index: This query embedding is then used to search the
    #      vector database (FAISS index in this case). FAISS efficiently calculates
    #      the "distance" (e.g., L2 distance, cosine similarity) between the query
    #      embedding and all the chunk embeddings stored in the index.
    #   3. Retrieving Top-K Chunks: The search returns the indices of the top-K
    #      most similar chunks (those with the smallest distance or highest similarity).
    #      These chunks are considered the most semantically relevant to the query.
    # These retrieved chunks are then passed to an LLM as context to generate an answer.

    # --- 4. Perform Retrieval ---
    sample_query = "What were the total revenues?"
    print(f"\n--- Performing Vector Retrieval for Query: \"{sample_query}\" ---")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Query embedding generation requires an OpenAI API key.")
        return

    try:
        llm = OpenAI(api_key=openai_api_key, timeout=20.0, max_retries=2) # Standard client
        print("OpenAI client initialized.")

        # Generate query embedding
        print("Generating embedding for the query...")
        # Using a newer model like text-embedding-3-large or text-embedding-3-small is recommended.
        # Ensure the dimensionality matches the one used for indexing (ada-002 was 1536).
        # text-embedding-3-large default is 3072, text-embedding-3-small is 1536.
        # If demo_09 used ada-002 (1536 dims), use text-embedding-3-small or specify dimensions.
        # For this demo, let's assume text-embedding-3-small for matching ada-002's common dim if needed.
        # Or, if the index was created with text-embedding-3-large, this is fine.
        # It's crucial that query embedding model matches document embedding model/dimensions.
        # Let's use text-embedding-ada-002 for consistency with typical FAISS index dimension of 1536.
        # If your index was built with a different model/dimension, adjust here.
        embedding_model = "text-embedding-ada-002" # Or "text-embedding-3-small" for 1536 dims
        
        embedding_response = llm.embeddings.create(
            input=sample_query,
            model=embedding_model
        )
        query_embedding = embedding_response.data[0].embedding
        query_embedding_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        print(f"Query embedding generated. Shape: {query_embedding_np.shape}")

        # Search the FAISS index
        top_k = 5 # Number of top results to retrieve
        print(f"Searching FAISS index for top {top_k} similar chunks...")
        distances, indices = faiss_index.search(query_embedding_np, top_k)
        
        print("\n--- Retrieval Results ---")
        if not indices[0].size:
            print("No results found.")
        else:
            for i in range(len(indices[0])):
                retrieved_chunk_index = indices[0][i]
                retrieved_distance = distances[0][i]

                if retrieved_chunk_index < 0 or retrieved_chunk_index >= len(chunks):
                    print(f"  Result {i+1}: Invalid index {retrieved_chunk_index} returned by FAISS. Skipping.")
                    continue

                retrieved_chunk = chunks[retrieved_chunk_index]
                chunk_id = retrieved_chunk.get('id', 'N/A')
                page_num = retrieved_chunk.get('page_number', 'N/A')
                chunk_text_snippet = retrieved_chunk.get('text', '')[:250] # First 250 chars

                print(f"  Result {i+1}:")
                print(f"    Chunk ID: {chunk_id}")
                print(f"    Page Number: {page_num}")
                # FAISS L2 distance is non-negative; smaller is better.
                # For cosine similarity, FAISS often returns 1 - cosine_sim for IndexFlatIP, so smaller is better.
                # If using IndexFlatL2, distance is Euclidean distance.
                print(f"    Similarity Score (Distance): {retrieved_distance:.4f}")
                print(f"    Text Snippet: \"{chunk_text_snippet}...\"")
                print("-" * 20)

    except Exception as e:
        print(f"An error occurred during vector retrieval: {e}")
        import traceback
        traceback.print_exc()
    print("----------------------------------------------------")

    print("\nVector retrieval demo complete.")

if __name__ == "__main__":
    main()
