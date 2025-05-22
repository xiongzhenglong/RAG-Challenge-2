# study/demo_12_bm25_retrieval.py

import json
import os
from pathlib import Path
import sys
import pickle # For loading the BM25 index object

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Note: BM25Ingestor is not directly needed here if we are just loading a pickled BM25Okapi object.
# However, the BM25Okapi object itself would have been created using a library like `rank_bm25`.

def main():
    """
    Demonstrates retrieving relevant text chunks using a pre-built BM25 index.
    This involves tokenizing a query, using the BM25 model to score chunks,
    and retrieving the top-scoring ones.
    """
    print("Starting BM25 retrieval demo...")

    # --- 1. Define Paths ---
    # Input chunked report (output of demo_07)
    input_chunked_report_dir = Path("study/chunked_reports_output/")
    input_chunked_filename = "report_for_serialization.json" # Assuming this name
    input_chunked_report_path = input_chunked_report_dir / input_chunked_filename

    # Input BM25 index (output of demo_10)
    bm25_index_dir = Path("study/bm25_indices/")
    bm25_index_filename = input_chunked_filename.replace(".json", ".bm25.pkl") # From demo_10
    bm25_index_path = bm25_index_dir / bm25_index_filename

    print(f"Input chunked report path: {input_chunked_report_path}")
    print(f"BM25 index path: {bm25_index_path}")

    # --- 2. Prepare Data (Load Chunked JSON and BM25 Index) ---
    if not input_chunked_report_path.exists():
        print(f"Error: Input chunked JSON file not found at {input_chunked_report_path}")
        print("Please ensure 'demo_07_text_splitting.py' has run successfully.")
        return
    if not bm25_index_path.exists():
        print(f"Error: BM25 index file not found at {bm25_index_path}")
        print("Please ensure 'demo_10_creating_bm25_index.py' has run successfully.")
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

    bm25_index = None
    try:
        with open(bm25_index_path, 'rb') as f:
            bm25_index = pickle.load(f) # This should be a BM25Okapi instance (or similar)
        print(f"Successfully loaded BM25 index from {bm25_index_path}")
        # Basic check if the loaded object has a 'get_scores' method
        if not hasattr(bm25_index, 'get_scores'):
            print("Error: Loaded BM25 index object does not have a 'get_scores' method.")
            print("Ensure it's a valid BM25 model (e.g., from rank_bm25 library).")
            return
    except Exception as e:
        print(f"Error loading BM25 index: {e}")
        return

    # --- 3. Understanding BM25 Retrieval ---
    # BM25 retrieval works as follows:
    #   1. Tokenize the Query: The input query string is broken down into individual
    #      words (tokens), often after lowercasing and basic cleaning.
    #   2. Score Documents (Chunks): The BM25 algorithm calculates a relevance score
    #      for each document (chunk) in the indexed corpus with respect to the
    #      tokenized query. This score is based on:
    #         - Term Frequency (TF): How often query terms appear in a document.
    #         - Inverse Document Frequency (IDF): How rare or common the query terms
    #           are across the entire corpus of documents. Rare terms get higher weight.
    #         - Document Length: BM25 penalizes very long documents that might match
    #           terms by chance, and normalizes for document length.
    #   3. Retrieve Top-N Chunks: The documents are ranked by their BM25 scores,
    #      and the top N highest-scoring documents are returned as the most relevant results.
    # This method is effective for keyword-based searches.

    # --- 4. Perform Retrieval ---
    sample_query = "What were the company's main risks?"
    print(f"\n--- Performing BM25 Retrieval for Query: \"{sample_query}\" ---")

    try:
        # Tokenize the query (simple lowercasing and splitting)
        tokenized_query = sample_query.lower().split()
        print(f"Tokenized query: {tokenized_query}")

        # Get scores from the BM25 index
        # The `get_scores` method of a BM25Okapi object (from rank_bm25 library)
        # takes the tokenized query and returns an array of scores, one for each document
        # in the corpus that the BM25 model was trained on.
        print("Calculating BM25 scores for all chunks...")
        doc_scores = bm25_index.get_scores(tokenized_query)
        
        if len(doc_scores) != len(chunks):
            print(f"Warning: Number of scores ({len(doc_scores)}) from BM25 "
                  f"does not match number of chunks ({len(chunks)}).")
            print("This indicates a mismatch between the indexed corpus and the loaded chunks.")
            # We can still proceed but results might be misaligned if the original corpus changed.

        # Get top N results
        top_k = 5
        # Sort indices by score in descending order
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        print(f"\n--- Top {top_k} BM25 Retrieval Results ---")
        if not top_indices:
            print("No results found or scores were all zero/negative.")
        else:
            for i, retrieved_chunk_index in enumerate(top_indices):
                # Ensure the index is valid for the loaded chunks list
                if retrieved_chunk_index < 0 or retrieved_chunk_index >= len(chunks):
                    print(f"  Result {i+1}: Invalid index {retrieved_chunk_index} from BM25 sort. Skipping.")
                    continue

                retrieved_chunk = chunks[retrieved_chunk_index]
                bm25_score = doc_scores[retrieved_chunk_index]
                
                chunk_id = retrieved_chunk.get('id', 'N/A')
                page_num = retrieved_chunk.get('page_number', 'N/A')
                chunk_text_snippet = retrieved_chunk.get('text', '')[:250] # First 250 chars

                print(f"  Result {i+1}:")
                print(f"    Chunk ID: {chunk_id}")
                print(f"    Page Number: {page_num}")
                print(f"    BM25 Score: {bm25_score:.4f}") # Higher is generally better
                print(f"    Text Snippet: \"{chunk_text_snippet}...\"")
                print("-" * 20)

    except Exception as e:
        print(f"An error occurred during BM25 retrieval: {e}")
        import traceback
        traceback.print_exc()
    print("----------------------------------------------------")

    print("\nBM25 retrieval demo complete.")

if __name__ == "__main__":
    main()
