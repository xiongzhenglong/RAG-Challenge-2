# study/demo_19_rephrasing_comparative_questions.py

import sys
import os
import json
from dotenv import load_dotenv

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api_requests import APIProcessor # Used for making LLM calls

# Load environment variables from .env file (especially OPENAI_API_KEY)
load_dotenv()

# --- Purpose of this Demo ---
# This script demonstrates how comparative questions involving multiple entities
# (e.g., companies) are rephrased into individual, focused questions for each entity.
#
# Why rephrase comparative questions for RAG?
#   - Targeted Retrieval: RAG systems are often optimized to retrieve information
#     for a single, clear query. Rephrasing allows the system to fetch the most
#     relevant documents for each entity separately.
#   - Contextual Clarity: When generating an answer, providing the LLM with context
#     specific to "AlphaCorp's R&D" and then separately for "BetaInc's R&D" can lead
#     to more accurate and well-supported individual facts.
#   - Simplified Answer Synthesis: Once individual pieces of information are gathered
#     for each entity, it's often easier for a subsequent LLM call (or even procedural logic)
#     to synthesize a comparative answer.
#
# The `APIProcessor.get_rephrased_questions` method internally uses an LLM
# (prompted by a structure similar to `src.prompts.RephrasedQuestionsPrompt`)
# to generate these entity-specific questions.
#
# IMPORTANT:
# An `OPENAI_API_KEY` must be set in your .env file in the project root.

def main():
    """
    Demonstrates rephrasing a comparative question into individual questions
    for each involved company using APIProcessor.
    """
    print("Starting comparative question rephrasing demo...")

    # --- 1. Check for API Key ---
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please ensure your API key is configured in the .env file in the project root.")
        return
    print("OPENAI_API_KEY found in environment.")

    # --- 2. Initialize APIProcessor ---
    try:
        api_processor = APIProcessor(provider="openai")
        print("APIProcessor initialized for OpenAI.")
    except Exception as e:
        print(f"Error initializing APIProcessor: {e}")
        return

    # --- 3. Define Sample Comparative Question and Companies ---
    sample_comparative_question = (
        "Compare the Research and Development (R&D) expenses of 'AlphaCorp' and 'BetaInc' for the fiscal year 2023. "
        "Which company invested more in R&D during this period?"
    )
    involved_companies = ["AlphaCorp", "BetaInc"]

    print("\n--- Original Comparative Question ---")
    print(f"  Question: \"{sample_comparative_question}\"")
    print(f"  Involved Companies: {involved_companies}")

    # --- 4. Rephrase the Comparative Question ---
    print("\nRequesting rephrased questions from LLM via APIProcessor...")
    print("(This involves an LLM call using a prompt similar to RephrasedQuestionsPrompt)...")

    try:
        # `get_rephrased_questions` uses an LLM to break down the comparative question
        # into individual questions for each company.
        rephrased_map = api_processor.get_rephrased_questions(
            original_question=sample_comparative_question,
            companies=involved_companies
        )

        print("\n--- Rephrased Questions Map ---")
        if rephrased_map and isinstance(rephrased_map, dict):
            for company, rephrased_q in rephrased_map.items():
                print(f"  For Company '{company}':")
                print(f"    Rephrased Question: \"{rephrased_q}\"")
        else:
            print("  Failed to get a valid rephrased questions map or the map was empty.")
            print(f"  Received: {rephrased_map}")

        # --- 5. Inspect API Response Data from the Rephrasing Call ---
        print("\n--- API Response Metadata (from api_processor.processor.response_data for rephrasing call) ---")
        if hasattr(api_processor.processor, 'response_data') and api_processor.processor.response_data:
            response_metadata = api_processor.processor.response_data
            
            if hasattr(response_metadata, 'model'):
                 print(f"  Model Used (from API): {response_metadata.model}")

            if hasattr(response_metadata, 'usage') and response_metadata.usage:
                usage_info = response_metadata.usage
                print(f"  Token Usage:")
                print(f"    Prompt Tokens: {usage_info.prompt_tokens}")
                print(f"    Completion Tokens: {usage_info.completion_tokens}")
                print(f"    Total Tokens: {usage_info.total_tokens}")
            else:
                print("  Token usage data not found in response_data.")
        else:
            print("  No additional response data found on api_processor.processor for the rephrasing call.")

    except Exception as e:
        print(f"\nAn error occurred during the question rephrasing process: {e}")
        print("This could be due to API key issues, network problems, LLM model errors,")
        print("or issues with the input question format.")
        import traceback
        traceback.print_exc()

    print("\nComparative question rephrasing demo complete.")

if __name__ == "__main__":
    main()
