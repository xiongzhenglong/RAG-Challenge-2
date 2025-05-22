# study/demo_21_simplified_pipeline_run.py

import sys
import os
import json
from pathlib import Path
import shutil
import pandas as pd # For creating the subset DataFrame
from dotenv import load_dotenv

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import Pipeline, RunConfig

# Load environment variables from .env file (especially OPENAI_API_KEY)
load_dotenv()

# --- Purpose of this Demo ---
# This script demonstrates a simplified end-to-end run of the `Pipeline` class.
# It sets up its own small dataset (one PDF, a subset CSV, and questions JSON),
# then executes the main processing steps:
#   1. PDF Parsing
#   2. Table Serialization
#   3. Post-processing (Merging, MD Export, Chunking, Vector DB Creation)
#   4. Question Processing (Answering)
# This provides a high-level overview of how the different components of the
# `financial-document-understanding` project integrate.
#
# IMPORTANT:
# - An `OPENAI_API_KEY` must be set in your .env file in the project root.
#   This is needed for table serialization, embedding generation, and question answering.
# - This demo creates a `study/pipeline_demo_data` directory. You might want to
#   delete this directory after running the demo.

def setup_pipeline_demo_data():
    """
    Sets up a small, isolated dataset for the pipeline demo.
    Returns the root path of the demo data if successful, None otherwise.
    """
    print("\n--- Setting up Pipeline Demo Data ---")
    demo_root_path = Path("study/pipeline_demo_data")
    pdf_reports_dir = demo_root_path / "pdf_reports"
    original_pdf_src = Path("data/test_set/pdf_reports/194000c9109c6fa628f1fed33b44ae4c2b8365f4.pdf")
    demo_pdf_name = "democorp_pipeline_report.pdf"
    demo_sha1 = "democorp_pipeline_report" # Stem of the PDF name, used as ID

    try:
        # Clear or create directories for a clean run
        if demo_root_path.exists():
            print(f"Found existing demo data directory: {demo_root_path}. Clearing it...")
            shutil.rmtree(demo_root_path)
        
        demo_root_path.mkdir(parents=True)
        pdf_reports_dir.mkdir(parents=True)
        print(f"Created demo data directories: {demo_root_path}, {pdf_reports_dir}")

        # Copy PDF
        if not original_pdf_src.exists():
            print(f"Error: Original source PDF not found at {original_pdf_src}")
            print("Please ensure the main dataset is available (e.g., via DVC).")
            return None
        shutil.copy(original_pdf_src, pdf_reports_dir / demo_pdf_name)
        print(f"Copied '{original_pdf_src.name}' to '{pdf_reports_dir / demo_pdf_name}'")

        # Create subset.csv
        subset_csv_path = demo_root_path / "subset_demo.csv"
        # Mimicking structure of the original subset.csv
        # (Only essential fields for the pipeline are strictly needed by `Pipeline` class itself)
        subset_data = {
            'sha1': [demo_sha1],
            'company_name': ['DemoCorp Pipeline'],
            'company_number': ['00000000'],
            'document_type': ['Annual Report'],
            'period_end_on': ['2023-12-31'],
            'retrieved_on': ['2024-01-01'],
            'source_url': ['http://example.com/report.pdf'],
            'lang': ['en']
        }
        pd.DataFrame(subset_data).to_csv(subset_csv_path, index=False)
        print(f"Created subset file: {subset_csv_path}")

        # Create questions.json
        questions_json_path = demo_root_path / "questions_demo.json"
        questions_data = [
            {"id": "dpq_1", "text": f"What were the total revenues for {subset_data['company_name'][0]}?", "kind": "number"},
            {"id": "dpq_2", "text": f"Who is the CEO of {subset_data['company_name'][0]}?", "kind": "name"}
        ]
        with open(questions_json_path, 'w', encoding='utf-8') as f:
            json.dump(questions_data, f, indent=2)
        print(f"Created questions file: {questions_json_path}")
        
        print("--- Pipeline Demo Data Setup Complete ---")
        return demo_root_path

    except Exception as e:
        print(f"Error setting up pipeline demo data: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_answers_file(answers_dir: Path, base_name: str):
    """Finds the answers file, accounting for potential numeric suffixes."""
    if (answers_dir / f"{base_name}.json").exists():
        return answers_dir / f"{base_name}.json"
    
    # Check for files like answers_demo_run_01.json, answers_demo_run_02.json etc.
    for i in range(1, 100): # Check a reasonable range
        suffixed_name = f"{base_name}_{i:02d}.json"
        if (answers_dir / suffixed_name).exists():
            return answers_dir / suffixed_name
    return None


def main():
    """
    Runs a simplified end-to-end pipeline.
    """
    print("Starting simplified pipeline run demo...")

    # --- Check for API Key ---
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("\nError: OPENAI_API_KEY environment variable is not set.")
        print("Please ensure your API key is configured in the .env file in the project root.")
        return
    print("\nOPENAI_API_KEY found in environment.")

    # --- Setup Demo Data ---
    demo_data_root = setup_pipeline_demo_data()
    if not demo_data_root:
        print("\nAborting demo due to errors in data setup.")
        return

    # --- Configure and Initialize Pipeline ---
    print("\n--- Configuring and Initializing Pipeline ---")
    # Using specific settings for a quick demo run
    run_config = RunConfig(
        use_serialized_tables=True,
        parent_document_retrieval=False, # Set to False if parent-child chunking wasn't explicitly done
        llm_reranking=True,
        config_suffix="_demo_run", # Suffix for output directories and files
        parallel_requests=1,       # Low parallelism for demo simplicity
        top_n_retrieval=3,
        llm_reranking_sample_size=3, # Rerank fewer items for speed
        submission_file=False        # Don't generate submission file for demo
    )
    print(f"RunConfig: {run_config}")

    try:
        pipeline = Pipeline(
            root_path=demo_data_root,
            subset_name="subset_demo.csv",
            questions_file_name="questions_demo.json",
            pdf_reports_dir_name="pdf_reports", # Relative to demo_data_root
            run_config=run_config
        )
        print("Pipeline initialized successfully.")
    except Exception as e:
        print(f"Error initializing Pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Run Pipeline Steps ---
    try:
        print("\n--- Step 1: Download Docling Models (if needed) ---")
        Pipeline.download_docling_models() # Static method call
        print("Docling model check/download complete.")

        print("\n--- Step 2: Parse PDF Reports (Sequential) ---")
        pipeline.parse_pdf_reports_sequential()
        print("PDF parsing complete.")
        # Output: demo_data_root / debug_data / parsed_reports_json_demo_run / democorp_pipeline_report.json

        print("\n--- Step 3: Serialize Tables ---")
        pipeline.serialize_tables(max_workers=1) # Low workers for demo
        print("Table serialization complete.")
        # Output: (Modifies files in) demo_data_root / debug_data / parsed_reports_json_demo_run /

        print("\n--- Step 4: Process Parsed Reports ---")
        # This includes: merging, MD export, chunking, vector DB (FAISS & BM25) creation
        pipeline.process_parsed_reports()
        print("Processing of parsed reports complete.")
        # Outputs:
        # - demo_data_root / debug_data / merged_reports_json_demo_run /
        # - demo_data_root / debug_data / reports_markdown_demo_run /
        # - demo_data_root / debug_data / chunked_reports_json_demo_run /
        # - demo_data_root / databases_demo_run / faiss_indices /
        # - demo_data_root / databases_demo_run / bm25_indices /

        print("\n--- Step 5: Process Questions ---")
        pipeline.process_questions()
        print("Question processing complete.")
        
        # --- Display Results ---
        print("\n--- Final Answers ---")
        # Answers are saved in demo_data_root / answers_demo_run.json (or with a numeric suffix)
        # The actual filename is determined by `_get_next_available_filename` in pipeline.py
        answers_base_name = f"answers{run_config.config_suffix}" # e.g., "answers_demo_run"
        
        # The answers file is directly under demo_data_root
        answers_file_path = find_answers_file(demo_data_root, answers_base_name)

        if answers_file_path and answers_file_path.exists():
            print(f"Loading answers from: {answers_file_path}")
            with open(answers_file_path, 'r', encoding='utf-8') as f:
                answers_content = json.load(f)
            print("Generated Answers (JSON content):")
            print(json.dumps(answers_content, indent=2, ensure_ascii=False))
        else:
            print(f"Could not find the answers JSON file. Expected base name: {answers_base_name}.json in {demo_data_root}")
            print(f"Please check the directory contents. Available files: {list(demo_data_root.iterdir())}")


    except Exception as e:
        print(f"\nAn error occurred during pipeline execution: {e}")
        import traceback
        traceback.print_exc()

    # --- Output and Cleanup Info ---
    print("\n--- Pipeline Run Complete ---")
    print(f"All outputs for this demo run are located under: {demo_data_root.resolve()}")
    print("Key subdirectories to inspect:")
    print(f"  - Parsed PDF JSON: {demo_data_root / 'debug_data' / f'parsed_reports_json{run_config.config_suffix}'}")
    print(f"  - Merged reports: {demo_data_root / 'debug_data' / f'merged_reports_json{run_config.config_suffix}'}")
    print(f"  - Chunked reports: {demo_data_root / 'debug_data' / f'chunked_reports_json{run_config.config_suffix}'}")
    print(f"  - Databases (FAISS, BM25): {demo_data_root / f'databases{run_config.config_suffix}'}")
    print(f"  - Final Answers: {demo_data_root} (look for '{answers_base_name}.json' or similar)")
    
    print("\nTo clean up, you can manually delete the entire directory:")
    print(f"  rm -rf {demo_data_root.resolve()}")

    print("\nSimplified pipeline run demo complete.")

if __name__ == "__main__":
    main()
