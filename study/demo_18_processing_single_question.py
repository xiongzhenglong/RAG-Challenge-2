# study/demo_18_processing_single_question.py

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.questions_processing import QuestionsProcessor
from src.api_requests import APIProcessor # For potential access to response_data

# Load environment variables from .env file (especially OPENAI_API_KEY)
load_dotenv()

# --- Purpose of this Demo ---
# This script demonstrates using the `QuestionsProcessor` class to automate
# the full RAG (Retrieval Augmented Generation) pipeline for answering a single question.
# `QuestionsProcessor` encapsulates:
#   - Identifying the target company/document.
#   - Retrieving relevant chunks (potentially using hybrid retrieval).
#   - Reranking chunks (if enabled).
#   - Generating an answer based on the context using an LLM, often with a
#     specific output schema.
#
# IMPORTANT:
# - An `OPENAI_API_KEY` must be set in your .env file in the project root.
# - This demo relies on files prepared by previous demos (chunked JSON, FAISS index).
#   The `ensure_demo_files_ready` function helps verify/prepare these.

def ensure_demo_files_ready(chunked_reports_dir: Path, demo_json_filename: str,
                            target_company_name: str, sha1_name_for_demo: str,
                            vector_dbs_dir: Path):
    """
    Ensures the necessary JSON and FAISS files are ready and correctly configured for the demo.
    Returns True if successful, False otherwise.
    """
    print("\n--- Ensuring Demo Files Are Ready ---")
    json_report_path = chunked_reports_dir / demo_json_filename
    expected_faiss_path = vector_dbs_dir / f"{sha1_name_for_demo}.faiss"

    # 1. Check and modify JSON metadata
    try:
        if not json_report_path.exists():
            print(f"Error: Chunked JSON report not found at {json_report_path}")
            print("Please run demo_07_text_splitting.py and ensure its output is available.")
            return False
        
        with open(json_report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        # Ensure metainfo exists
        if 'metainfo' not in report_data:
            report_data['metainfo'] = {}
        
        # Set/verify company_name and sha1_name
        report_data['metainfo']['company_name'] = target_company_name
        report_data['metainfo']['sha1_name'] = sha1_name_for_demo
        
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        print(f"Verified/Updated metadata in {json_report_path}:")
        print(f"  - Company Name: '{target_company_name}'")
        print(f"  - SHA1 Name (for FAISS): '{sha1_name_for_demo}'")

    except Exception as e:
        print(f"Error preparing JSON metadata for {json_report_path}: {e}")
        return False

    # 2. Check for FAISS index
    if not expected_faiss_path.exists():
        print(f"Error: Expected FAISS index not found at {expected_faiss_path}")
        print(f"Please ensure 'demo_09_creating_vector_db.py' was run and its output "
              f"'{sha1_name_for_demo}.faiss' (or copied from 'demo_report.faiss' and renamed) exists.")
        return False
    print(f"Found expected FAISS index at {expected_faiss_path}.")
    
    print("--- Demo Files Ready ---")
    return True

def main():
    """
    Demonstrates processing a single question using QuestionsProcessor.
    """
    print("Starting single question processing demo...")

    # --- Define Paths & Config ---
    chunked_reports_dir = Path("study/chunked_reports_output/")
    vector_dbs_dir = Path("study/vector_dbs/")
    demo_json_filename = "report_for_serialization.json" # Used throughout demos
    target_company_name = "TestCorp Inc."
    sha1_name_for_demo = "report_for_serialization" # Stem of the FAISS file

    # --- Check for API Key ---
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("\nError: OPENAI_API_KEY environment variable is not set.")
        print("Please ensure your API key is configured in the .env file in the project root.")
        return
    print("\nOPENAI_API_KEY found in environment.")

    # --- Ensure Demo Files are Ready ---
    if not ensure_demo_files_ready(chunked_reports_dir, demo_json_filename,
                                   target_company_name, sha1_name_for_demo, vector_dbs_dir):
        print("\nAborting demo due to issues with required files.")
        return

    # --- Initialize QuestionsProcessor ---
    print("\nInitializing QuestionsProcessor...")
    try:
        # Enable LLM reranking and Parent Document Retrieval for a comprehensive demo
        processor = QuestionsProcessor(
            vector_db_dir=vector_dbs_dir,
            documents_dir=chunked_reports_dir,
            llm_reranking=True,
            parent_document_retrieval=True, # Assumes parent-child chunking if applicable
            api_provider="openai"
        )
        print("QuestionsProcessor initialized successfully.")
    except Exception as e:
        print(f"Error initializing QuestionsProcessor: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Process a Single Question ---
    question = f"What were the total revenues for {target_company_name} in the last fiscal year?"
    # Schema defines the expected type/structure of the answer.
    # Options: "name", "names", "number", "boolean", "default" (for general text)
    schema = "number" 

    print(f"\n--- Processing Question ---")
    print(f"  Question: \"{question}\"")
    print(f"  Expected Schema: \"{schema}\"")
    print("(This involves retrieval, reranking, and LLM answer generation - may take time)...")

    try:
        answer_dict = processor.process_question(question=question, schema=schema)

        print("\n--- Question Processing Result ---")
        if answer_dict:
            print("  Full Answer Dictionary (from Pydantic model, e.g., AnswerSchemaNumber):")
            print(json.dumps(answer_dict, indent=2, ensure_ascii=False))
            
            print("\n  Key Extracted Information:")
            print(f"    Final Answer: {answer_dict.get('final_answer', 'N/A')}")
            print(f"    Step-by-Step Analysis:\n      {answer_dict.get('step_by_step_analysis', 'N/A').replace(os.linesep, os.linesep + '      ')}")
            
            relevant_pages = answer_dict.get('relevant_pages')
            if relevant_pages:
                print(f"    Relevant Pages: {', '.join(map(str, relevant_pages))}")
            
            # --- Optional: Inspect APIProcessor's last response_data ---
            # This gives insight into the final LLM call made for answer generation.
            # processor.api_processor should be the instance used by QuestionsProcessor.
            if hasattr(processor, 'api_processor') and \
               hasattr(processor.api_processor, 'processor') and \
               hasattr(processor.api_processor.processor, 'response_data') and \
               processor.api_processor.processor.response_data:
                
                response_metadata = processor.api_processor.processor.response_data
                print("\n  Metadata from the Final LLM Call (Answer Generation):")
                if hasattr(response_metadata, 'model'):
                    print(f"    Model Used: {response_metadata.model}")
                if hasattr(response_metadata, 'usage') and response_metadata.usage:
                    usage_info = response_metadata.usage
                    print(f"    Token Usage: Prompt={usage_info.prompt_tokens}, Completion={usage_info.completion_tokens}, Total={usage_info.total_tokens}")
            else:
                print("\n  No additional response data found on QuestionsProcessor's APIProcessor instance.")
        else:
            print("  Processing did not return an answer dictionary.")

    except Exception as e:
        print(f"\nAn error occurred during question processing: {e}")
        import traceback
        traceback.print_exc()
    print("----------------------------------------------------")

    # --- Cleanup Note ---
    print("\n--- Demo Cleanup Reminder ---")
    print(f"  - The JSON file '{chunked_reports_dir / demo_json_filename}' may have been modified by 'ensure_demo_files_ready'.")
    print("  You may want to revert these changes or delete the copied/modified files if you rerun demos or for cleanup.")


    print("\nSingle question processing demo complete.")

if __name__ == "__main__":
    main()
