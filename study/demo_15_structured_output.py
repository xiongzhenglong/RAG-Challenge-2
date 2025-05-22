# study/demo_15_structured_output.py

import sys
import os
import json
from pathlib import Path # Not strictly used but good practice for path handling
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api_requests import APIProcessor

# Load environment variables from .env file (especially OPENAI_API_KEY)
load_dotenv()

# --- Purpose of this Demo ---
# This script demonstrates how to use the `APIProcessor` class to request
# and receive structured JSON output from an OpenAI Language Model (LLM).
# By providing a Pydantic model as the desired `response_format`, the LLM
# is instructed (often via function calling or JSON mode) to generate output
# that conforms to the schema of that Pydantic model. `APIProcessor` then
# automatically parses this JSON output into a Python dictionary.
#
# IMPORTANT:
# An `OPENAI_API_KEY` must be set in your .env file in the project root.
# Structured output capabilities may depend on the specific LLM model used
# (e.g., "gpt-4o-mini", "gpt-4-turbo" support JSON mode well).

# --- 1. Define Pydantic Model for Structured Output ---
class SimpleResponse(BaseModel):
    answer: str = Field(description="The direct answer to the question.")
    confidence_score: float = Field(description="A score from 0.0 to 1.0 indicating confidence.")
    related_topics: list[str] = Field(description="A list of related topics.")

def main():
    """
    Demonstrates sending a request to an OpenAI LLM and receiving
    a structured JSON response parsed according to a Pydantic model.
    """
    print("Starting structured output (JSON with Pydantic) demo...")

    # --- 2. Check for API Key ---
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please ensure your API key is configured in the .env file in the project root.")
        return
    print("OPENAI_API_KEY found in environment.")

    # --- 3. Initialize APIProcessor ---
    try:
        api_processor = APIProcessor(provider="openai")
        print("APIProcessor initialized for OpenAI.")
    except Exception as e:
        print(f"Error initializing APIProcessor: {e}")
        return

    # --- 4. Define Request Components ---
    question = "What is the capital of France and what are two related topics regarding its history?"
    
    # System content guides the LLM. When using Pydantic models for response_format with OpenAI,
    # the APIProcessor (or underlying OpenAI client) typically appends the JSON schema of the
    # Pydantic model to the system message or uses tools/function-calling to enforce the structure.
    # This system_content is a general instruction.
    system_content = (
        "You are an AI assistant. Answer the user's question. You must provide a direct answer, "
        "a confidence score (from 0.0 to 1.0), and a list of two related topics. "
        "Format your response according to the provided schema."
    )

    print("\n--- Request Details ---")
    print(f"  Question: \"{question}\"")
    print(f"  System Content Hint (schema is also sent by APIProcessor):\n    \"{system_content}\"")
    print(f"  Expected Pydantic Schema: {SimpleResponse.__name__}")

    # --- 5. Send Request to LLM for Structured Output ---
    llm_model_name = "gpt-4o-mini" # Or "gpt-4-turbo", "gpt-3.5-turbo" (check model's JSON mode support)
    print(f"\nSending request to OpenAI model: {llm_model_name} for structured output...")

    try:
        # `is_structured=True` and `response_format=SimpleResponse` (the Pydantic model class)
        # signals APIProcessor to configure the request for structured JSON output.
        response_dict = api_processor.send_message(
            model=llm_model_name,
            system_content=system_content,
            human_content=question,
            temperature=0.1,
            is_structured=True,
            response_format=SimpleResponse # Pass the Pydantic model class
        )

        print("\n--- LLM Response ---")
        print(f"  Original Question: {question}")

        if response_dict and isinstance(response_dict, dict):
            print("\n  Structured LLM Response (parsed as dictionary by APIProcessor):")
            print(json.dumps(response_dict, indent=2))

            print("\n  Accessing individual fields from the dictionary:")
            print(f"    Answer: {response_dict.get('answer')}")
            print(f"    Confidence Score: {response_dict.get('confidence_score')}")
            related = response_dict.get('related_topics', [])
            print(f"    Related Topics: {', '.join(related) if related else 'N/A'}")
        elif response_dict: # If not a dict but something was returned
            print("\n  Received a non-dictionary response (unexpected for structured output):")
            print(f"    Type: {type(response_dict)}")
            print(f"    Content: {response_dict}")
        else:
            print("\n  Failed to get a structured response or response was None.")

        # --- 6. Inspect Raw API Response Data ---
        print("\n--- API Response Metadata (from api_processor.processor.response_data) ---")
        if hasattr(api_processor.processor, 'response_data') and api_processor.processor.response_data:
            response_metadata = api_processor.processor.response_data
            if hasattr(response_metadata, 'model'):
                 print(f"  Model Used (from API): {response_metadata.model}")
            if hasattr(response_metadata, 'usage') and response_metadata.usage:
                usage_info = response_metadata.usage
                print(f"  Token Usage: Prompt={usage_info.prompt_tokens}, Completion={usage_info.completion_tokens}, Total={usage_info.total_tokens}")
            else:
                print("  Token usage data not found in response_data.")
        else:
            print("  No additional response data found on api_processor.processor.")

    except Exception as e:
        print(f"\nAn error occurred during the API request for structured output: {e}")
        print("This could be due to issues with the LLM's ability to conform to the schema,")
        print("API key problems, network issues, or model limitations.")
        import traceback
        traceback.print_exc()

    print("\nStructured output demo complete.")

if __name__ == "__main__":
    main()
