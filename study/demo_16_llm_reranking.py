# study/demo_16_llm_reranking.py

import sys
import os
import json
from dotenv import load_dotenv

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.reranking import LLMReranker

# Load environment variables from .env file (especially OPENAI_API_KEY)
load_dotenv()

# --- Purpose of this Demo ---
# This script demonstrates LLM-based reranking of search results.
# Initial retrieval methods (like vector search or BM25) provide a set of candidate
# documents. LLM-based reranking uses a more powerful Language Model to assess
# the relevance of these candidates to the query in a more nuanced way.
# The LLM typically assigns a relevance score, which can then be combined with
# the initial retrieval score to produce a final, refined ranking.
# This helps to improve the precision of the top results shown to the user or
# passed to a final answer generation step.
#
# IMPORTANT:
# An `OPENAI_API_KEY` must be set in your .env file in the project root,
# as `LLMReranker` uses an LLM (e.g., OpenAI's GPT models) to perform the reranking.

def main():
    """
    Demonstrates LLM-based reranking of a sample list of documents
    against a sample query.
    """
    print("Starting LLM-based reranking demo...")

    # --- 1. Prepare Sample Data ---
    sample_query = "Tell me about the company's sustainability efforts and impact on local communities."

    sample_documents = [
        {
            "id": "doc1",
            "text": "Our company is committed to reducing its carbon footprint. We have invested in renewable energy sources and aim for carbon neutrality by 2030. Our annual report details these initiatives.",
            "distance": 0.85, # Initial similarity score (higher is better)
            "page": 5
        },
        {
            "id": "doc2",
            "text": "The new product line, launched in Q3, has exceeded sales expectations. Marketing strategies focused on digital channels and influencer collaborations.",
            "distance": 0.72,
            "page": 12
        },
        {
            "id": "doc3",
            "text": "We actively engage with local communities through various programs, including educational scholarships and environmental cleanup drives. Last year, we dedicated over 1000 volunteer hours to these causes.",
            "distance": 0.90,
            "page": 8
        },
        {
            "id": "doc4",
            "text": "Financial highlights for the fiscal year include a 15% increase in revenue. The primary drivers were strong performance in the North American market and cost optimization measures.",
            "distance": 0.65,
            "page": 2
        }
    ]

    print("\n--- Sample Query ---")
    print(f"  \"{sample_query}\"")

    print("\n--- Documents Before Reranking (Sorted by initial 'distance' score desc) ---")
    # Sort by initial distance for clarity before reranking
    sorted_initial_docs = sorted(sample_documents, key=lambda x: x['distance'], reverse=True)
    for doc in sorted_initial_docs:
        print(f"  ID: {doc['id']}, Initial Distance (Similarity): {doc['distance']:.2f}, "
              f"Page: {doc.get('page', 'N/A')}, Text Snippet: \"{doc['text'][:100]}...\"")

    # --- 2. Check for API Key ---
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("\nError: OPENAI_API_KEY environment variable is not set.")
        print("Please ensure your API key is configured in the .env file in the project root.")
        return
    print("\nOPENAI_API_KEY found in environment.")

    # --- 3. Initialize LLMReranker ---
    try:
        # LLMReranker uses an LLM (defaulting to OpenAI's models)
        # to assess the relevance of each document to the query.
        reranker = LLMReranker()
        print("LLMReranker initialized.")
    except Exception as e:
        print(f"Error initializing LLMReranker: {e}")
        return

    # --- 4. Perform Reranking ---
    print("\nPerforming reranking using LLM...")
    print("(This may take some time as it involves LLM calls for document batches)...")

    try:
        # `rerank_documents` processes documents in batches, sends them to the LLM
        # with the query, and gets relevance scores.
        # `llm_weight` controls how much the LLM's relevance score influences the
        # final combined score relative to the initial retrieval score ('distance').
        reranked_documents = reranker.rerank_documents(
            query=sample_query,
            documents=sample_documents,
            documents_batch_size=2, # Process 2 documents per LLM call for this demo
            llm_weight=0.7          # Give LLM's assessment more weight
        )

        print("\n--- Documents After Reranking (Sorted by 'combined_score' desc) ---")
        if not reranked_documents:
            print("  No documents returned after reranking.")
        else:
            # The reranked_documents list should already be sorted by 'combined_score'
            for doc in reranked_documents:
                print(f"  ID: {doc['id']}")
                print(f"    Original Distance (Similarity): {doc.get('distance', 'N/A'):.2f}")
                print(f"    LLM Relevance Score: {doc.get('relevance_score', 'N/A'):.2f}")
                print(f"    Combined Score: {doc.get('combined_score', 'N/A'):.2f}")
                print(f"    Page: {doc.get('page', 'N/A')}")
                print(f"    Text Snippet: \"{doc.get('text', '')[:100]}...\"")
                print(f"    LLM Reasoning (if available): {doc.get('reasoning', 'N/A')}")
                print("-" * 20)
            
            print("\n  Note on Ordering:")
            print("  The order of documents may have changed significantly from the initial ranking.")
            print("  Documents with high LLM relevance scores can be promoted even if their initial")
            print("  'distance' (similarity) score was not the highest, and vice-versa.")
            print("  The 'combined_score' reflects this new, refined ranking.")

    except Exception as e:
        print(f"\nAn error occurred during the reranking process: {e}")
        print("This could be due to API key issues, network problems, LLM model errors,")
        print("or issues with the input data format.")
        import traceback
        traceback.print_exc()

    print("\nLLM-based reranking demo complete.")

if __name__ == "__main__":
    main()
