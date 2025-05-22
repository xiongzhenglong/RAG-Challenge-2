# study/demo_14_openai_api_request.py

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api_requests import APIProcessor

# Load environment variables from .env file (especially OPENAI_API_KEY)
load_dotenv()

# --- Purpose of this Demo ---
# This script demonstrates how to use the `APIProcessor` class to interact
# with an OpenAI Language Model (LLM) like GPT-4o-mini.
# It shows a basic Request-Augmented Generation (RAG) pattern where a question
# is answered based on provided context.
#
# IMPORTANT:
# An `OPENAI_API_KEY` must be set in your .env file in the project root
# for this script to work. Refer to `study/demo_01_project_setup.py`
# for instructions on setting up your .env file and API keys.

def main():
    """
    Demonstrates sending a request to an OpenAI LLM using APIProcessor
    with a sample question and RAG context.
    """
    print("Starting OpenAI API request demo...")

    # --- 1. Check for API Key ---
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please ensure your API key is configured in the .env file in the project root.")
        print("Refer to 'study/demo_01_project_setup.py' for guidance.")
        return

    print("OPENAI_API_KEY found in environment.")

    # --- 2. Initialize APIProcessor ---
    try:
        # APIProcessor can be configured for different providers (e.g., "openai", "google", "ibm")
        # It handles the underlying client initialization and request formatting.
        api_processor = APIProcessor(provider="openai")
        print("APIProcessor initialized for OpenAI.")
    except Exception as e:
        print(f"Error initializing APIProcessor: {e}")
        return

    # --- 3. Define Request Components ---
    question = "What is the main product of ExampleCorp?"
    rag_context = (
        "ExampleCorp is a leading provider of advanced widget solutions. "
        "Their flagship product, the 'SuperWidget', is known for its "
        "efficiency and reliability. ExampleCorp also offers consulting services "
        "for widget integration."
    )

    # System content guides the LLM's behavior and tone.
    system_content = (
        "You are a helpful assistant. Your task is to answer the user's question "
        "based *only* on the provided context. If the answer cannot be found within "
        "the context, you must explicitly state 'Information not found in the provided context.' "
        "Do not make up information or use external knowledge."
    )

    # Human content combines the context and the specific question.
    human_content = f"Context:\n---\n{rag_context}\n---\n\nQuestion: {question}"

    print("\n--- Request Details ---")
    print(f"  System Content (Instructions to LLM):\n    \"{system_content}\"")
    print(f"\n  Human Content (Context + Question):\n    Context: \"{rag_context[:100]}...\""
          f"\n    Question: \"{question}\"")

    # --- 4. Send Request to LLM ---
    # We use "gpt-4o-mini" as it's a fast and cost-effective model suitable for many tasks.
    # Other models like "gpt-4-turbo" or "gpt-3.5-turbo" can also be used.
    # Temperature controls randomness: 0.1 makes the output more deterministic.
    llm_model_name = "gpt-4o-mini"
    print(f"\nSending request to OpenAI model: {llm_model_name}...")

    try:
        response = api_processor.send_message(
            model=llm_model_name,
            system_content=system_content,
            human_content=human_content,
            temperature=0.1,
            # max_tokens=100 # Optionally control max output length
        )

        print("\n--- LLM Response ---")
        print(f"  Question: {question}")
        print(f"  Provided RAG Context: \"{rag_context}\"")
        print(f"\n  LLM's Answer:\n    \"{response}\"")

        # --- 5. Inspect Response Data (Token Usage, Model Info) ---
        # The `api_processor.processor.response_data` attribute (if available on the
        # specific processor like `OpenAIProcessor`) often stores the raw response object
        # from the API, which includes metadata like token usage.
        print("\n--- API Response Metadata (from api_processor.processor.response_data) ---")
        if hasattr(api_processor.processor, 'response_data') and api_processor.processor.response_data:
            response_metadata = api_processor.processor.response_data
            
            # Print the model used (as reported by the API)
            if hasattr(response_metadata, 'model'):
                 print(f"  Model Used (from API): {response_metadata.model}")

            # Print token usage if available
            if hasattr(response_metadata, 'usage') and response_metadata.usage:
                usage_info = response_metadata.usage
                print(f"  Token Usage:")
                print(f"    Prompt Tokens: {usage_info.prompt_tokens}")
                print(f"    Completion Tokens: {usage_info.completion_tokens}")
                print(f"    Total Tokens: {usage_info.total_tokens}")
            else:
                print("  Token usage data not found in response_data.")
            
            # You can print the whole object too, but it can be verbose
            # print(f"  Full response_data object:\n{response_metadata}")
        else:
            print("  No additional response data found on api_processor.processor.")

    except Exception as e:
        print(f"\nAn error occurred during the API request: {e}")
        print("This could be due to various reasons such as:")
        print("  - Incorrect API key or insufficient credits.")
        print("  - Network connectivity issues.")
        print("  - Issues with the requested model (e.g., availability, rate limits).")
        print("  - Problems with the input data format or length.")
        import traceback
        traceback.print_exc()

    print("\nOpenAI API request demo complete.")

if __name__ == "__main__":
    main()
