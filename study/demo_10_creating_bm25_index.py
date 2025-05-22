# study/demo_10_creating_bm25_index.py

import json
import os
from pathlib import Path
import sys
import pickle # For saving the BM25 index object

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import BM25Ingestor

def main():
    """
    Demonstrates creating a BM25 sparse retrieval index from the text chunks
    of a processed report.
    """
    print("Starting BM25 index creation demo...")

    # --- 1. Define Paths ---
    # Input is the chunked report (output of demo_07)
    input_chunked_report_dir = Path("study/chunked_reports_output/")
    input_chunked_filename = "report_for_serialization.json" # Assuming this name
    input_chunked_report_path = input_chunked_report_dir / input_chunked_filename

    # Output directory for the BM25 index
    bm25_output_dir = Path("study/bm25_indices/")
    # Determine BM25 index filename
    bm25_index_filename = input_chunked_filename.replace(".json", ".bm25.pkl")
    bm25_index_path = bm25_output_dir / bm25_index_filename

    print(f"Input chunked report path: {input_chunked_report_path}")
    print(f"BM25 index output directory: {bm25_output_dir}")
    print(f"BM25 index will be saved to: {bm25_index_path}")

    # --- 2. Prepare Input Data (Load Chunked JSON) ---
    if not input_chunked_report_path.exists():
        print(f"Error: Input chunked JSON file not found at {input_chunked_report_path}")
        print("Please ensure 'demo_07_text_splitting.py' has run successfully.")
        return

    all_chunk_texts = []
    try:
        with open(input_chunked_report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        if 'content' in report_data and 'chunks' in report_data['content'] and report_data['content']['chunks']:
            # Extract just the text content of each chunk
            all_chunk_texts = [chunk['text'] for chunk in report_data['content']['chunks'] if chunk.get('text')]
            if not all_chunk_texts:
                 print("No text content found in any chunks. Cannot create BM25 index.")
                 return
            print(f"Successfully loaded chunked JSON. Found {len(all_chunk_texts)} text chunks.")
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

    # --- 3. Understanding BM25 (Okapi BM25) ---
    # BM25 (Best Matching 25) is a ranking function used by search engines to estimate
    # the relevance of documents to a given search query. It's a bag-of-words retrieval
    # function that ranks a set of documents based on the query terms appearing in each document.
    #
    # Key Characteristics:
    #   - Term Frequency-based: Unlike dense vector retrieval (which uses semantic embeddings),
    #     BM25 relies on the frequency of query terms within the documents (chunks, in our case)
    #     and the inverse document frequency (IDF) of those terms across the entire corpus.
    #   - Sparse Retrieval: It operates on sparse vector representations of text (e.g., TF-IDF vectors),
    #     focusing on exact keyword matches and their statistical importance.
    #   - Keyword Matching: It excels at finding documents that contain the exact keywords from the query.
    #
    # Contrast with Dense Vector Retrieval (e.g., using FAISS with embeddings):
    #   - Dense Retrieval: Captures semantic similarity. Can find relevant documents even if they
    #     don't use the exact query keywords but discuss similar concepts. Uses dense vector embeddings.
    #   - BM25 (Sparse Retrieval): Relies on keyword overlap. May miss documents that are semantically
    #     related but use different terminology.
    #
    # Common Use Cases:
    #   - Baseline: Often used as a strong baseline for information retrieval tasks.
    #   - Hybrid Retrieval: Frequently combined with dense retrieval methods. The idea is to leverage
    #     the strengths of both: BM25's keyword matching precision and dense retrieval's semantic understanding.
    #     Scores from both systems can be combined (e.g., using reciprocal rank fusion) to produce a final ranking.
    #
    # The `BM25Ingestor` likely uses a library like `rank_bm25` to create the index.

    # --- 4. Create BM25 Index ---
    print("\nInitializing BM25Ingestor and creating the BM25 index...")

    try:
        ingestor = BM25Ingestor() # This might initialize the BM25 model (e.g., BM25Okapi from rank_bm25)
        print("BM25Ingestor initialized.")

        print(f"Creating BM25 index from {len(all_chunk_texts)} text chunks...")
        # The `create_bm25_index` method would take the list of text chunks
        # and fit the BM25 model to this corpus.
        bm25_index = ingestor.create_bm25_index(all_chunk_texts)

        if not bm25_index: # Basic check
            print("Error: BM25 index creation failed or returned an invalid object.")
            return
            
        print(f"BM25 index created successfully.")
        # Note: The BM25 index object itself (e.g., a BM25Okapi instance) doesn't
        # typically have a direct '.ntotal' like FAISS. Its "size" is implicit
        # in the corpus it was trained on.

        # Create the output directory if it doesn't exist
        bm25_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {bm25_output_dir}")

        # Save the BM25 index to disk using pickle
        # BM25 models (like those from rank_bm25) are typically Python objects
        # and can be serialized using pickle.
        print(f"Saving BM25 index to: {bm25_index_path}...")
        with open(bm25_index_path, 'wb') as f:
            pickle.dump(bm25_index, f)
        print(f"BM25 index successfully saved to {bm25_index_path}")

    except Exception as e:
        print(f"An error occurred during BM25 index creation or saving: {e}")
        import traceback
        traceback.print_exc()
    print("----------------------------------------------------")

    print("\nBM25 index creation demo complete.")
    print("The generated .pkl file contains the BM25 model/index, ready for keyword-based searches.")

if __name__ == "__main__":
    main()
