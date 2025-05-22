# study/demo_13_understanding_rag_prompts.py

import sys
import os
import inspect # To get source code of Pydantic models
import json # For pretty printing Pydantic schema

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prompts import (
    RephrasedQuestionsPrompt,
    AnswerWithRAGContextNamePrompt,
    AnswerWithRAGContextNumberPrompt,
    AnswerWithRAGContextBooleanPrompt,
    AnswerWithRAGContextNamesPrompt,
    ComparativeAnswerPrompt,
    RerankingPrompt,
    RetrievalRankingSingleBlock,  # Pydantic model
    RetrievalRankingMultipleBlocks # Pydantic model
)

def get_attribute_safely(prompt_obj, attr_name):
    """Safely gets an attribute from a prompt object, returning a default string if not found."""
    return getattr(prompt_obj, attr_name, "N/A (attribute not found)")

def main():
    """
    Displays and explains key prompt structures from src/prompts.py.
    These prompts are fundamental to how the RAG system interacts with LLMs
    for various tasks like question rephrasing, answer generation, and reranking.
    """
    print("--- Understanding RAG Prompts from src/prompts.py ---")
    print("This script displays the structure and purpose of various prompts used in the RAG pipeline.\n")

    # --- 1. RephrasedQuestionsPrompt ---
    print("\n--- 1. RephrasedQuestionsPrompt ---")
    print(f"  Purpose: To generate multiple rephrased versions of an initial user query. "
          f"This helps in retrieving a broader set of potentially relevant documents.")
    print(f"  Instruction:\n{get_attribute_safely(RephrasedQuestionsPrompt, 'instruction')}")
    print(f"\n  User Prompt (template):\n{get_attribute_safely(RephrasedQuestionsPrompt, 'user_prompt')}")
    print(f"\n  Pydantic Schema (source code for expected output structure):\n"
          f"{get_attribute_safely(RephrasedQuestionsPrompt, 'pydantic_schema')}")
    print(f"\n  Example (how the prompt might be filled and its expected output):\n"
          f"{get_attribute_safely(RephrasedQuestionsPrompt, 'example')}")
    print("-" * 50)

    # --- 2. AnswerWithRAGContextNamePrompt ---
    print("\n--- 2. AnswerWithRAGContextNamePrompt ---")
    print(f"  Purpose: To generate a concise answer (typically a name or short phrase) "
          f"based on a specific question and provided RAG context.")
    print(f"  Instruction:\n{get_attribute_safely(AnswerWithRAGContextNamePrompt, 'instruction')}")
    print(f"\n  User Prompt (template):\n{get_attribute_safely(AnswerWithRAGContextNamePrompt, 'user_prompt')}")
    print(f"\n  Pydantic Schema (source code):\n"
          f"{get_attribute_safely(AnswerWithRAGContextNamePrompt, 'pydantic_schema')}")
    print(f"\n  Example:\n{get_attribute_safely(AnswerWithRAGContextNamePrompt, 'example')}")
    print("-" * 50)

    # --- 3. AnswerWithRAGContextNumberPrompt ---
    print("\n--- 3. AnswerWithRAGContextNumberPrompt ---")
    print(f"  Purpose: Similar to NamePrompt, but specifically for extracting numerical answers.")
    print(f"  Instruction:\n{get_attribute_safely(AnswerWithRAGContextNumberPrompt, 'instruction')}")
    print(f"\n  User Prompt (template):\n{get_attribute_safely(AnswerWithRAGContextNumberPrompt, 'user_prompt')}")
    print(f"\n  Pydantic Schema (source code):\n"
          f"{get_attribute_safely(AnswerWithRAGContextNumberPrompt, 'pydantic_schema')}")
    print(f"\n  Example:\n{get_attribute_safely(AnswerWithRAGContextNumberPrompt, 'example')}")
    print("-" * 50)

    # --- 4. AnswerWithRAGContextBooleanPrompt ---
    print("\n--- 4. AnswerWithRAGContextBooleanPrompt ---")
    print(f"  Purpose: For questions requiring a boolean (Yes/No) answer, along with supporting evidence from the context.")
    print(f"  Instruction:\n{get_attribute_safely(AnswerWithRAGContextBooleanPrompt, 'instruction')}")
    print(f"\n  User Prompt (template):\n{get_attribute_safely(AnswerWithRAGContextBooleanPrompt, 'user_prompt')}")
    print(f"\n  Pydantic Schema (source code):\n"
          f"{get_attribute_safely(AnswerWithRAGContextBooleanPrompt, 'pydantic_schema')}")
    print(f"\n  Example:\n{get_attribute_safely(AnswerWithRAGContextBooleanPrompt, 'example')}")
    print("-" * 50)

    # --- 5. AnswerWithRAGContextNamesPrompt ---
    print("\n--- 5. AnswerWithRAGContextNamesPrompt ---")
    print(f"  Purpose: To extract a list of names or short phrases in response to a question, based on RAG context.")
    print(f"  Instruction:\n{get_attribute_safely(AnswerWithRAGContextNamesPrompt, 'instruction')}")
    print(f"\n  User Prompt (template):\n{get_attribute_safely(AnswerWithRAGContextNamesPrompt, 'user_prompt')}")
    print(f"\n  Pydantic Schema (source code):\n"
          f"{get_attribute_safely(AnswerWithRAGContextNamesPrompt, 'pydantic_schema')}")
    print(f"\n  Example:\n{get_attribute_safely(AnswerWithRAGContextNamesPrompt, 'example')}")
    print("-" * 50)

    # --- 6. ComparativeAnswerPrompt ---
    print("\n--- 6. ComparativeAnswerPrompt ---")
    print(f"  Purpose: To generate answers for comparative questions involving multiple entities or aspects, using RAG context.")
    print(f"  Instruction:\n{get_attribute_safely(ComparativeAnswerPrompt, 'instruction')}")
    print(f"\n  User Prompt (template):\n{get_attribute_safely(ComparativeAnswerPrompt, 'user_prompt')}")
    print(f"\n  Pydantic Schema (source code):\n"
          f"{get_attribute_safely(ComparativeAnswerPrompt, 'pydantic_schema')}")
    print(f"\n  Example:\n{get_attribute_safely(ComparativeAnswerPrompt, 'example')}")
    print("-" * 50)

    # --- 7. RerankingPrompt ---
    print("\n--- 7. RerankingPrompt ---")
    print(f"  Purpose: To have an LLM rerank an initial set of retrieved document chunks based on their relevance to the query. "
          f"This helps to refine the search results before final answer generation.")
    # RerankingPrompt has specific system prompts instead of a single 'instruction'.
    print(f"  System Prompt (Rerank Single Block):\n"
          f"{get_attribute_safely(RerankingPrompt, 'system_prompt_rerank_single_block')}")
    print(f"\n  System Prompt (Rerank Multiple Blocks):\n"
          f"{get_attribute_safely(RerankingPrompt, 'system_prompt_rerank_multiple_blocks')}")
    # User prompt for reranking is typically constructed dynamically with the query and context.
    # The Pydantic schemas are for the expected output structure.
    print(f"\n  Pydantic Schema (Single Block - source code):\n"
          f"{get_attribute_safely(RerankingPrompt, 'pydantic_schema_single_block')}")
    print(f"\n  Pydantic Schema (Multiple Blocks - source code):\n"
          f"{get_attribute_safely(RerankingPrompt, 'pydantic_schema_multiple_blocks')}")
    print(f"\n  Example (Single Block - how the prompt might be filled and its expected output):\n"
          f"{get_attribute_safely(RerankingPrompt, 'example_single_block')}")
    print(f"\n  Example (Multiple Blocks - how the prompt might be filled and its expected output):\n"
          f"{get_attribute_safely(RerankingPrompt, 'example_multiple_blocks')}")
    print("-" * 50)

    # --- 8. RetrievalRankingSingleBlock (Pydantic Model) ---
    print("\n--- 8. RetrievalRankingSingleBlock (Pydantic Model) ---")
    print(f"  Purpose: Defines the expected JSON output structure when an LLM reranks a single retrieved text block. "
          f"It includes fields for relevance, confidence, and reasoning.")
    try:
        # Print the source code of the Pydantic model
        schema_source = inspect.getsource(RetrievalRankingSingleBlock)
        print(f"  Pydantic Model Source Code:\n{schema_source}")
        # Alternatively, print the JSON schema:
        # print(f"  Pydantic Model JSON Schema:\n{json.dumps(RetrievalRankingSingleBlock.model_json_schema(), indent=2)}")
    except TypeError:
        print("  Could not retrieve source code for RetrievalRankingSingleBlock (likely not a class/module).")
    except Exception as e:
        print(f"  Error retrieving schema for RetrievalRankingSingleBlock: {e}")
    print("-" * 50)

    # --- 9. RetrievalRankingMultipleBlocks (Pydantic Model) ---
    print("\n--- 9. RetrievalRankingMultipleBlocks (Pydantic Model) ---")
    print(f"  Purpose: Defines the expected JSON output structure when an LLM reranks multiple retrieved text blocks. "
          f"It typically contains a list of objects, each conforming to a structure similar to RetrievalRankingSingleBlock.")
    try:
        # Print the source code of the Pydantic model
        schema_source = inspect.getsource(RetrievalRankingMultipleBlocks)
        print(f"  Pydantic Model Source Code:\n{schema_source}")
        # Alternatively, print the JSON schema:
        # print(f"  Pydantic Model JSON Schema:\n{json.dumps(RetrievalRankingMultipleBlocks.model_json_schema(), indent=2)}")
    except TypeError:
        print("  Could not retrieve source code for RetrievalRankingMultipleBlocks (likely not a class/module).")
    except Exception as e:
        print(f"  Error retrieving schema for RetrievalRankingMultipleBlocks: {e}")
    print("-" * 50)

    print("\nPrompt exploration complete.")

if __name__ == "__main__":
    main()
