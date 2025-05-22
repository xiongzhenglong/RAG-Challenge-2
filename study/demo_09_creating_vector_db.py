# study/demo_09_creating_vector_db.py

import json
import os
from pathlib import Path
import sys
import faiss # Explicitly import faiss to show it's being used
import numpy as np # VectorDBIngestor._create_vector_db might expect numpy arrays

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import VectorDBIngestor

def main():
    """
    Demonstrates creating a FAISS vector database from the text chunks
    of a processed report. This involves generating embeddings for all chunks
    and then indexing them with FAISS.
    """
    print("Starting FAISS vector database creation demo...")

    # --- 1. Define Paths ---
    # Input is the chunked report (output of demo_07)
    input_chunked_report_dir = Path("study/chunked_reports_output/")
    input_chunked_filename = "report_for_serialization.json" # Assuming this name
    input_chunked_report_path = input_chunked_report_dir / input_chunked_filename

    # Output directory for the FAISS index
    vector_db_output_dir = Path("study/vector_dbs/")
    # Determine FAISS index filename (e.g., based on input or a generic name)
    # For simplicity, using a generic name for this demo.
    # In a real system, this might be derived from the report's SHA1 or ID.
    faiss_index_filename = "demo_report.faiss"
    faiss_index_path = vector_db_output_dir / faiss_index_filename

    print(f"Input chunked report path: {input_chunked_report_path}")
    print(f"Vector DB output directory: {vector_db_output_dir}")
    print(f"FAISS index will be saved to: {faiss_index_path}")

    # --- 2. Prepare Input Data (Load Chunked JSON) ---
    if not input_chunked_report_path.exists():
        print(f"Error: Input chunked JSON file not found at {input_chunked_report_path}")
        print("Please ensure 'demo_07_text_splitting.py' has run successfully.")
        return

    all_chunks_data = []
    try:
        with open(input_chunked_report_path, 'r', encoding='utf-8') as f:
            chunked_data = json.load(f)
        
        if 'content' in chunked_data and 'chunks' in chunked_data['content'] and chunked_data['content']['chunks']:
            all_chunks_data = chunked_data['content']['chunks']
            if not all_chunks_data:
                 print("No chunks found in the loaded JSON file. Cannot create vector DB.")
                 return
            print(f"Successfully loaded chunked JSON. Found {len(all_chunks_data)} chunks.")
        else:
            print("Error: 'content' or 'chunks' not found in the loaded JSON structure.")
            print("Please ensure the input file is correctly formatted (output of demo_07).")
            return
            
    except json.JSONDecodeError:
        print(f"Error: Could not decode the JSON file at {input_chunked_report_path}.")
        return
    except Exception as e:
        print(f"An error occurred while loading the JSON file: {e}")
        return

    # --- 3. Understanding Vector Databases and FAISS ---
    # Vector Databases:
    #   - These are specialized databases designed to store, manage, and efficiently
    #     search through large collections of vector embeddings.
    #   - In RAG, when a user query is converted to an embedding, the vector database
    #     is used to quickly find the most similar (semantically relevant) text chunk
    #     embeddings from the knowledge base.
    #
    # FAISS (Facebook AI Similarity Search):
    #   - FAISS is an open-source library developed by Facebook AI for efficient
    #     similarity search and clustering of dense vectors.
    #   - It's not a full-fledged database system itself but provides the core indexing
    #     and search algorithms that can be integrated into such systems or used directly.
    #   - FAISS supports various indexing methods suitable for different trade-offs
    #     between search speed, accuracy, and memory usage.
    #
    # API Key for Embeddings:
    #   - The first step to creating a vector DB is getting embeddings for all text chunks.
    #   - As highlighted in demo_08, this requires an embedding model, often accessed via API
    #     (e.g., OpenAI). Ensure your `OPENAI_API_KEY` is set in your environment.

    # --- 4. Create FAISS Index ---
    print("\nInitializing VectorDBIngestor to generate embeddings and create FAISS index...")

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Embeddings generation (a prerequisite for FAISS index) requires an OpenAI API key.")
        print("Please set it up (see demo_01_project_setup.py) and try again.")
        return

    try:
        ingestor = VectorDBIngestor()
        print("VectorDBIngestor initialized.")

        # Extract text from all chunks
        all_texts = [chunk['text'] for chunk in all_chunks_data if chunk.get('text')]
        if not all_texts:
            print("No text content found in any of the chunks. Cannot generate embeddings.")
            return
        
        print(f"Generating embeddings for {len(all_texts)} text chunks...")
        print("(This may take some time and involve multiple API calls)...")
        embeddings_list = ingestor._get_embeddings(all_texts) # Returns a list of lists (embeddings)

        if not embeddings_list or not isinstance(embeddings_list, list) or not all(isinstance(e, list) for e in embeddings_list):
            print("Error: Embeddings generation did not return the expected list of embedding vectors.")
            return
        
        # Convert list of lists to a 2D NumPy array for FAISS
        # Ensure all embeddings have the same dimension (e.g., 1536 for text-embedding-ada-002)
        try:
            embeddings_np = np.array(embeddings_list).astype('float32')
        except ValueError as ve:
            print(f"Error converting embeddings to NumPy array: {ve}")
            print("This might happen if embeddings have inconsistent dimensions.")
            # Optional: print dimensions of a few embeddings to debug
            # for i, emb in enumerate(embeddings_list[:3]): print(f"Emb {i} len: {len(emb)}")
            return

        print(f"Embeddings generated. Shape of embedding matrix: {embeddings_np.shape}")

        print("Creating FAISS index...")
        # `_create_vector_db` in `VectorDBIngestor` likely handles FAISS index creation.
        # It might look something like:
        #   d = embeddings_np.shape[1]  # Dimensionality of embeddings
        #   index = faiss.IndexFlatL2(d) # Simple L2 distance index
        #   index.add(embeddings_np)
        # We'll call the ingestor's method which encapsulates this.
        index = ingestor._create_vector_db(embeddings_np) # Pass the NumPy array

        if not index or not hasattr(index, 'ntotal'): # Basic check for a FAISS index object
            print("Error: FAISS index creation failed or returned an invalid object.")
            return
            
        print(f"FAISS index created successfully. Index contains {index.ntotal} vectors.")

        # Create the output directory if it doesn't exist
        vector_db_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {vector_db_output_dir}")

        # Save the FAISS index to disk
        print(f"Saving FAISS index to: {faiss_index_path}...")
        faiss.write_index(index, str(faiss_index_path))
        print(f"FAISS index successfully saved to {faiss_index_path}")

    except Exception as e:
        print(f"An error occurred during FAISS index creation or saving: {e}")
        import traceback
        traceback.print_exc()
    print("----------------------------------------------------")

    print("\nFAISS vector database creation demo complete.")
    print("The generated .faiss file, along with a corresponding mapping file (usually created by VectorDBIngestor.ingest_reports),")
    print("would be used by the RAG system for similarity searches.")

if __name__ == "__main__":
    main()
