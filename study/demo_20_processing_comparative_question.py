# study/demo_20_processing_comparative_question.py

import sys
import os
import json
from pathlib import Path
import shutil
from dotenv import load_dotenv
import faiss
import numpy as np

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.questions_processing import QuestionsProcessor
from src.api_requests import APIProcessor # For APIProcessor within QuestionsProcessor
from src.ingestion import VectorDBIngestor # For creating FAISS indices

# Load environment variables from .env file (especially OPENAI_API_KEY)
load_dotenv()

# --- Purpose of this Demo ---
# This script demonstrates how `QuestionsProcessor` handles comparative questions
# involving multiple entities (companies). The process typically involves:
#   1. Rephrasing: The comparative question is broken down into individual,
#      focused questions for each entity (e.g., "What were AlphaCorp's revenues?",
#      "What were BetaInc's revenues?"). (Handled internally by QuestionsProcessor)
#   2. Individual Answering: Each rephrased question is processed using the RAG
#      pipeline (retrieval, reranking, LLM answer generation) to find the specific
#      information for that entity.
#   3. Final Synthesis: The individual answers are then provided to an LLM with the
#      original comparative question to synthesize a final comparative answer.
#      (e.g., "BetaInc had higher revenues than AlphaCorp.").
#
# IMPORTANT:
# - An `OPENAI_API_KEY` must be set in your .env file in the project root.
# - This demo creates and modifies JSON files and FAISS indices specifically for
#   this comparative scenario.

def prepare_comparative_demo_data(
    base_chunked_json_template_path: Path,
    chunked_reports_dir: Path,
    vector_dbs_dir: Path,
    company_name: str,
    revenue_text: str,
    overwrite: bool = True # Set to False to skip if files exist
):
    """
    Prepares a dedicated chunked JSON report and FAISS index for a company.
    Modifies metadata and injects specific revenue information into a chunk.
    """
    print(f"\n--- Preparing Demo Data for: {company_name} ---")
    company_id = company_name.lower()
    target_json_path = chunked_reports_dir / f"{company_id}.json"
    target_faiss_path = vector_dbs_dir / f"{company_id}.faiss"

    if not overwrite and target_json_path.exists() and target_faiss_path.exists():
        print(f"Data for {company_name} already exists and overwrite is False. Skipping preparation.")
        return True

    # 1. Create and Modify JSON Report
    try:
        if not base_chunked_json_template_path.exists():
            print(f"Error: Base template JSON not found at {base_chunked_json_template_path}")
            return False
        
        chunked_reports_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(base_chunked_json_template_path, target_json_path)
        print(f"Copied template to {target_json_path}")

        with open(target_json_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        report_data['metainfo'] = {
            "company_name": company_name,
            "sha1_name": company_id # Links to the FAISS index
        }
        
        # Inject revenue info into the first text chunk for simplicity
        if report_data.get('content', {}).get('chunks'):
            # Ensure there's at least one chunk
            if not report_data['content']['chunks']:
                 report_data['content']['chunks'].append({"id": "chunk_0", "type": "content", "page_number": 1, "text": ""})

            # Prepend or replace text of the first chunk
            original_text = report_data['content']['chunks'][0].get('text', '')
            report_data['content']['chunks'][0]['text'] = f"{revenue_text} " + original_text
            print(f"Modified first chunk in {target_json_path} with revenue info.")
        else:
            print(f"Warning: No chunks found in {target_json_path} to modify. Creating a dummy chunk.")
            report_data['content']['chunks'] = [{"id": "chunk_0", "type": "content", "page_number": 1, "text": revenue_text}]


        with open(target_json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        print(f"Saved modified JSON for {company_name} to {target_json_path}")

    except Exception as e:
        print(f"Error preparing JSON data for {company_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. Create FAISS Index for the Modified JSON
    try:
        vector_dbs_dir.mkdir(parents=True, exist_ok=True)
        ingestor = VectorDBIngestor() # Needs OPENAI_API_KEY for embeddings

        chunks_for_faiss = report_data.get('content', {}).get('chunks', [])
        if not chunks_for_faiss:
            print(f"No chunks found in {target_json_path} to create FAISS index. Skipping.")
            return False # Cannot proceed without chunks

        chunk_texts = [chunk['text'] for chunk in chunks_for_faiss if chunk.get('text')]
        if not chunk_texts:
            print(f"No text content in chunks for {company_name}. Creating FAISS index with dummy data might fail or be meaningless.")
            # Create a dummy entry to avoid faiss errors with empty data, though this is not ideal.
            chunk_texts = ["dummy text for faiss index"]


        print(f"Generating embeddings for {len(chunk_texts)} chunks for {company_name}...")
        embeddings_list = ingestor._get_embeddings(chunk_texts)
        
        if not embeddings_list:
            print(f"Failed to generate embeddings for {company_name}. Skipping FAISS index creation.")
            return False

        embeddings_np = np.array(embeddings_list).astype('float32')
        
        faiss_index = ingestor._create_vector_db(embeddings_np)
        faiss.write_index(faiss_index, str(target_faiss_path))
        print(f"Created and saved FAISS index for {company_name} to {target_faiss_path}")

    except Exception as e:
        print(f"Error creating FAISS index for {company_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """
    Demonstrates processing a comparative question using QuestionsProcessor.
    """
    print("Starting comparative question processing demo...")

    # --- Define Paths & Config ---
    base_template_json_path = Path("study/chunked_reports_output/report_for_serialization.json")
    chunked_reports_dir = Path("study/comparative_demo_data/chunked_reports/")
    vector_dbs_dir = Path("study/comparative_demo_data/vector_dbs/")
    
    company1_name = "AlphaCorp"
    company1_revenue_text = "AlphaCorp's total revenue in 2023 was $500 million."
    
    company2_name = "BetaInc"
    company2_revenue_text = "BetaInc's total revenue in 2023 was $750 million."

    # --- Check for API Key ---
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("\nError: OPENAI_API_KEY environment variable is not set.")
        print("Please ensure your API key is configured in the .env file in the project root.")
        return
    print("\nOPENAI_API_KEY found in environment.")

    # --- Prepare Demo Data for Both Companies ---
    # Ensure the base template from previous demos exists
    if not base_template_json_path.exists():
        print(f"Error: Base template JSON '{base_template_json_path}' not found.")
        print("Please run demo_07_text_splitting.py to generate 'report_for_serialization.json'.")
        return

    print("Preparing data for AlphaCorp...")
    if not prepare_comparative_demo_data(base_template_json_path, chunked_reports_dir, vector_dbs_dir,
                                         company1_name, company1_revenue_text, overwrite=True):
        print(f"\nFailed to prepare data for {company1_name}. Aborting demo.")
        return
    
    print("\nPreparing data for BetaInc...")
    if not prepare_comparative_demo_data(base_template_json_path, chunked_reports_dir, vector_dbs_dir,
                                         company2_name, company2_revenue_text, overwrite=True):
        print(f"\nFailed to prepare data for {company2_name}. Aborting demo.")
        return

    # --- Initialize QuestionsProcessor ---
    print("\nInitializing QuestionsProcessor...")
    try:
        # For comparative questions, QuestionsProcessor needs access to the directories
        # where all relevant company reports (JSONs and FAISS indices) are stored.
        processor = QuestionsProcessor(
            vector_db_dir=vector_dbs_dir,
            documents_dir=chunked_reports_dir,
            llm_reranking=True, # Recommended for better context selection
            parent_document_retrieval=False, # Keep false if not specifically set up
            api_provider="openai",
            new_challenge_pipeline=False # Using False to avoid subset_path dependency for this demo
        )
        print("QuestionsProcessor initialized successfully.")
    except Exception as e:
        print(f"Error initializing QuestionsProcessor: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Process a Comparative Question ---
    comparative_question = (
        f"Which company had higher total revenue in 2023: '{company1_name}' or '{company2_name}'? "
        "Provide the revenue for each and state which one was higher."
    )
    # Schema "comparative" guides the LLM to provide a comparative answer.
    # The final answer format will be influenced by `ComparativeAnswerPrompt`.
    schema = "comparative" 

    print(f"\n--- Processing Comparative Question ---")
    print(f"  Question: \"{comparative_question}\"")
    print(f"  Expected Schema: \"{schema}\"")
    print("(This involves rephrasing, individual retrieval, reranking, and final LLM synthesis - may take significant time)...")

    try:
        answer_dict = processor.process_question(
            question=comparative_question, 
            schema=schema
            # companies_for_comparison can be explicitly passed if not reliably extracted from question:
            # companies_for_comparison=[company1_name, company2_name] 
        )

        print("\n--- Comparative Question Processing Result ---")
        if answer_dict:
            print("  Full Answer Dictionary (from Pydantic model, e.g., ComparativeAnswer):")
            print(json.dumps(answer_dict, indent=2, ensure_ascii=False))
            
            print("\n  Key Extracted Information:")
            print(f"    Final Comparative Answer: {answer_dict.get('final_answer', 'N/A')}")
            print(f"    Step-by-Step Analysis:\n      {answer_dict.get('step_by_step_analysis', 'N/A').replace(os.linesep, os.linesep + '      ')}")
            
            # Details of answers for individual companies might be in 'individual_answers'
            # if the ComparativeAnswerPrompt schema includes it and QuestionsProcessor populates it.
            individual_answers = answer_dict.get('individual_answers', [])
            if individual_answers:
                print("\n  Details for Individual Companies (from rephrased questions):")
                for ans_item in individual_answers:
                    print(f"    - Company: {ans_item.get('company_name', 'N/A')}")
                    print(f"      Answer: {ans_item.get('answer', 'N/A')}")
                    print(f"      Context Snippet: {ans_item.get('context_snippet', 'N/A')[:100]}...")


            # --- Optional: Inspect APIProcessor's last response_data for the final synthesis step ---
            if hasattr(processor, 'api_processor') and \
               hasattr(processor.api_processor, 'processor') and \
               hasattr(processor.api_processor.processor, 'response_data') and \
               processor.api_processor.processor.response_data:
                
                response_metadata = processor.api_processor.processor.response_data
                print("\n  Metadata from the Final LLM Call (Comparative Synthesis):")
                if hasattr(response_metadata, 'model'):
                    print(f"    Model Used: {response_metadata.model}")
                if hasattr(response_metadata, 'usage') and response_metadata.usage:
                    usage_info = response_metadata.usage
                    print(f"    Token Usage: Prompt={usage_info.prompt_tokens}, Completion={usage_info.completion_tokens}, Total={usage_info.total_tokens}")
            else:
                print("\n  No additional response data found on QuestionsProcessor's APIProcessor for the final synthesis.")
        else:
            print("  Processing did not return an answer dictionary for the comparative question.")

    except Exception as e:
        print(f"\nAn error occurred during comparative question processing: {e}")
        import traceback
        traceback.print_exc()
    print("----------------------------------------------------")

    # --- Cleanup Note ---
    print("\n--- Demo Cleanup Reminder ---")
    print(f"  - This demo created/modified files in '{chunked_reports_dir}' and '{vector_dbs_dir}'.")
    print(f"    Specifically: '{company1_name.lower()}.json', '{company1_name.lower()}.faiss', "
          f"'{company2_name.lower()}.json', '{company2_name.lower()}.faiss'.")
    print("  You may want to delete these directories or their contents for cleanup,")
    print("  especially if you plan to rerun this demo or other demos that might conflict.")


    print("\nComparative question processing demo complete.")

if __name__ == "__main__":
    main()
