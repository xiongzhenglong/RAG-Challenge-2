# study/demo_17_hybrid_retrieval.py

import sys
import os
import json
from pathlib import Path
import shutil
from dotenv import load_dotenv

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval import HybridRetriever

# Load environment variables from .env file (especially OPENAI_API_KEY)
load_dotenv()

# --- Purpose of this Demo ---
# This script demonstrates hybrid retrieval using the `HybridRetriever`.
# Hybrid retrieval combines:
#   1. Initial Candidate Retrieval: Usually dense vector search (like FAISS)
#      and/or sparse retrieval (like BM25) to fetch a larger set of potentially
#      relevant document chunks. (HybridRetriever internally uses VectorRetriever)
#   2. LLM-based Reranking: A Language Model then reranks these candidates
#      to improve the precision of the top results.
# This approach leverages the efficiency of traditional retrieval methods and the
# nuanced understanding of LLMs for relevance assessment.
#
# IMPORTANT:
# - An `OPENAI_API_KEY` must be set in your .env file in the project root.
#   This is needed by:
#     - `VectorRetriever` (inside `HybridRetriever`) for generating query embeddings.
#     - `LLMReranker` (inside `HybridRetriever`) for the reranking step.
# - This demo modifies a JSON file and renames/copies a FAISS index for demonstration
#   purposes. Note the cleanup instructions.

def prepare_demo_files(chunked_reports_dir, demo_json_filename, target_company_name,
                       vector_dbs_dir, demo_faiss_filename_original, demo_faiss_filename_expected):
    """
    Prepares files for the demo:
    1. Modifies the metadata of the chunked JSON report.
    2. Renames/copies the FAISS index to match the JSON's expected naming convention.
    Returns True if successful, False otherwise.
    """
    print("\n--- Preparing Demo Files ---")
    json_report_path = chunked_reports_dir / demo_json_filename
    original_faiss_path = vector_dbs_dir / demo_faiss_filename_original
    expected_faiss_path = vector_dbs_dir / demo_faiss_filename_expected

    # 1. Modify JSON metadata
    try:
        if not json_report_path.exists():
            print(f"Error: Chunked JSON report not found at {json_report_path}")
            return False
        
        with open(json_report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        # Ensure metainfo exists
        if 'metainfo' not in report_data:
            report_data['metainfo'] = {}
        
        report_data['metainfo']['company_name'] = target_company_name
        # The sha1_name should match the stem of the FAISS file for VectorRetriever
        report_data['metainfo']['sha1_name'] = expected_faiss_path.stem 
        
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        print(f"Successfully modified metadata in {json_report_path}:")
        print(f"  - Set company_name to: '{target_company_name}'")
        print(f"  - Set sha1_name to: '{expected_faiss_path.stem}'")

    except Exception as e:
        print(f"Error modifying JSON metadata for {json_report_path}: {e}")
        return False

    # 2. Rename/Copy FAISS index
    try:
        if not original_faiss_path.exists():
            print(f"Warning: Original FAISS index '{original_faiss_path}' not found. "
                  f"If '{expected_faiss_path}' already exists from a previous run, demo might still work.")
            # If expected already exists, we assume it's correctly set up
            if expected_faiss_path.exists():
                 print(f"Found expected FAISS index at '{expected_faiss_path}'. Proceeding.")
                 return True # Allow to proceed if expected file is already there
            return False # Original not found and expected not found

        if original_faiss_path == expected_faiss_path:
            print(f"FAISS index '{original_faiss_path}' already has the expected name. No action needed.")
        elif expected_faiss_path.exists():
            print(f"Expected FAISS index '{expected_faiss_path}' already exists. Overwriting for demo consistency.")
            shutil.copy2(original_faiss_path, expected_faiss_path) # copy2 preserves metadata
            print(f"Copied '{original_faiss_path}' to '{expected_faiss_path}' (overwrite).")
        else:
            shutil.copy2(original_faiss_path, expected_faiss_path)
            print(f"Copied '{original_faiss_path}' to '{expected_faiss_path}'.")
            
    except Exception as e:
        print(f"Error renaming/copying FAISS index from {original_faiss_path} to {expected_faiss_path}: {e}")
        return False
    
    print("--- Demo File Preparation Complete ---")
    return True

def main():
    """
    Demonstrates hybrid retrieval (vector search + LLM reranking)
    using HybridRetriever.
    """
    print("Starting hybrid retrieval demo...")

    # --- Define Paths & Config ---
    chunked_reports_dir = Path("study/chunked_reports_output/")
    vector_dbs_dir = Path("study/vector_dbs/")
    demo_json_filename = "report_for_serialization.json" # From demo_07
    demo_faiss_filename_original = "demo_report.faiss"   # From demo_09
    # HybridRetriever (via VectorRetriever) expects FAISS filename to match JSON filename stem
    demo_faiss_filename_expected = demo_json_filename.replace(".json", ".faiss") 
    target_company_name = "TestCorp Inc." # This name will be injected into JSON metadata

    # --- Check for API Key ---
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("\nError: OPENAI_API_KEY environment variable is not set.")
        print("Please ensure your API key is configured in the .env file in the project root.")
        return
    print("\nOPENAI_API_KEY found in environment.")

    # --- Prepare Demo Files (Modify JSON, Rename FAISS) ---
    if not prepare_demo_files(chunked_reports_dir, demo_json_filename, target_company_name,
                              vector_dbs_dir, demo_faiss_filename_original, demo_faiss_filename_expected):
        print("\nAborting demo due to errors in file preparation.")
        return

    # --- Initialize HybridRetriever ---
    try:
        # HybridRetriever orchestrates VectorRetriever and LLMReranker.
        # It needs paths to directories containing the chunked JSON reports and FAISS indices.
        retriever = HybridRetriever(
            vector_db_dir=vector_dbs_dir,
            documents_dir=chunked_reports_dir
        )
        print("\nHybridRetriever initialized successfully.")
    except Exception as e:
        print(f"\nError initializing HybridRetriever: {e}")
        return

    # --- Perform Hybrid Retrieval ---
    sample_query = "What are the key financial highlights and sustainability efforts?"
    print(f"\n--- Performing Hybrid Retrieval ---")
    print(f"  Target Company Name: \"{target_company_name}\"")
    print(f"  Sample Query: \"{sample_query}\"")
    print("(This involves vector search, then LLM reranking - may take time)...")

    try:
        # `retrieve_by_company_name` first finds the report for the company,
        # then performs vector search on its chunks, and finally reranks a subset.
        results = retriever.retrieve_by_company_name(
            company_name=target_company_name,
            query=sample_query,
            llm_reranking_sample_size=5, # No. of top vector search results to rerank
            top_n=3                      # Final number of results to return
        )

        print("\n--- Hybrid Retrieval Results (Top 3 after reranking) ---")
        if not results:
            print("  No results retrieved. This could be due to:")
            print("    - No matching report found for the company name.")
            print("    - No relevant chunks found by vector search.")
            print("    - Issues during the reranking process.")
        else:
            for i, chunk_info in enumerate(results):
                print(f"  Result {i+1}:")
                print(f"    Chunk ID: {chunk_info.get('id', 'N/A')}")
                print(f"    Page: {chunk_info.get('page_number', 'N/A')}")
                # The 'score' here is the combined_score from LLMReranker
                print(f"    Final Score (Combined): {chunk_info.get('score', 'N/A'):.4f}")
                print(f"    Text Snippet: \"{chunk_info.get('text', '')[:200]}...\"")
                # If LLMReranker's output includes original_score (from vector search)
                # and llm_relevance_score, you could print them too.
                # This depends on what HybridRetriever passes through.
                if 'original_score' in chunk_info: # Assuming vector search score
                     print(f"    Original Vector Search Score: {chunk_info['original_score']:.4f}")
                if 'llm_relevance_score' in chunk_info:
                     print(f"    LLM Relevance Score: {chunk_info['llm_relevance_score']:.4f}")
                print("-" * 20)

    except FileNotFoundError as fnf_error:
        print(f"\nError during retrieval: {fnf_error}")
        print("This often means a required JSON or FAISS file was not found for the target company.")
        print("Please check paths and ensure 'prepare_demo_files' ran correctly.")
    except Exception as e:
        print(f"\nAn error occurred during hybrid retrieval: {e}")
        import traceback
        traceback.print_exc()
    print("----------------------------------------------------")

    # --- Cleanup Note ---
    print("\n--- Demo Cleanup Reminder ---")
    print(f"  - The JSON file '{chunked_reports_dir / demo_json_filename}' has been modified.")
    print(f"    Its 'metainfo' now contains 'company_name': '{target_company_name}' and 'sha1_name': '{demo_faiss_filename_expected.stem}'.")
    print(f"  - The FAISS index file '{vector_dbs_dir / demo_faiss_filename_original}' might have been copied to "
          f"'{vector_dbs_dir / demo_faiss_filename_expected}'.")
    print("  You may want to revert these changes or delete the copied/modified files if you rerun demos or for cleanup.")

    print("\nHybrid retrieval demo complete.")

if __name__ == "__main__":
    main()
