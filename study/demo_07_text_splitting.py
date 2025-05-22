# study/demo_07_text_splitting.py

import json
import os
import shutil
from pathlib import Path
import sys

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.text_splitter import TextSplitter

def main():
    """
    Demonstrates text splitting (chunking) of merged reports using TextSplitter.
    This process prepares the content for Retrieval Augmented Generation (RAG) systems
    by breaking it into smaller, manageable pieces and optionally including
    serialized tables as separate chunks.
    """
    print("Starting text splitting (chunking) demo...")

    # --- 1. Define Paths ---
    # Input merged report (output of demo_05)
    input_merged_reports_dir = Path("study/merged_reports_output/")
    input_merged_filename = "report_for_serialization.json" # Assuming this name
    input_merged_full_path = input_merged_reports_dir / input_merged_filename

    # Input serialized tables report (output of demo_04)
    # This is used to extract serialized table data as separate chunks.
    serialized_tables_input_dir = Path("study/temp_serialization_data/")
    serialized_tables_filename = "report_for_serialization.json" # Assuming this name
    serialized_tables_full_path = serialized_tables_input_dir / serialized_tables_filename

    # Output directory for the chunked report
    chunked_output_dir = Path("study/chunked_reports_output/")
    # TextSplitter saves the output with the same name as the input merged file
    chunked_output_path = chunked_output_dir / input_merged_filename

    print(f"Input merged report directory: {input_merged_reports_dir}")
    print(f"Expected merged JSON file: {input_merged_full_path}")
    print(f"Input serialized tables directory: {serialized_tables_input_dir}")
    print(f"Expected serialized tables JSON file: {serialized_tables_full_path}")
    print(f"Chunked report output directory: {chunked_output_dir}")
    print(f"Expected chunked output file: {chunked_output_path}")

    # --- 2. Prepare Input Data ---
    if not input_merged_full_path.exists():
        print(f"Error: Input merged JSON file not found at {input_merged_full_path}")
        print("Please ensure 'demo_05_merging_reports.py' has run successfully.")
        return
    if not serialized_tables_full_path.exists():
        print(f"Error: Serialized tables JSON file not found at {serialized_tables_full_path}")
        print("Please ensure 'demo_04_serializing_tables.py' has run successfully.")
        return

    # Create the chunked output directory if it doesn't exist
    chunked_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured chunked output directory exists: {chunked_output_dir}")

    # --- 3. Understanding Text Splitting (Chunking) ---
    # Text splitting is crucial for RAG systems due to several reasons:
    #   - Context Window Limits: LLMs have a maximum limit on the amount of text
    #     they can process at once (the context window). Full documents often exceed this.
    #   - Targeted Retrieval: Smaller chunks allow for more precise retrieval. When a user
    #     asks a question, the system can find the most relevant chunks of text instead
    #     of an entire document, leading to more focused and accurate answers.
    #   - Efficiency: Processing smaller chunks is faster and less resource-intensive.
    #
    # `TextSplitter` in this project likely uses a strategy like Langchain's
    # `RecursiveCharacterTextSplitter`. This method tries to split text based on a
    # hierarchy of characters (e.g., "\n\n", "\n", " ", "") to keep semantically
    # related pieces of text together as much as possible.
    #   - Chunk Size: Defines the desired maximum size of each chunk (e.g., in characters or tokens).
    #   - Chunk Overlap: Defines a small overlap between consecutive chunks. This helps
    #     maintain context across chunk boundaries, ensuring that information isn't lost.
    #
    # This demo also demonstrates a key feature: incorporating serialized tables as
    # distinct chunks. Instead of just splitting the textual representation of tables
    # (which might be a flat Markdown string in the merged report), `TextSplitter`
    # can be configured to take the rich, LLM-generated `information_blocks` from
    # `TableSerializer` (demo_04) and treat each block as a separate, context-rich chunk.
    # This provides high-quality, structured information about tables to the RAG system.

    # --- 4. Perform Splitting ---
    print("\nInitializing TextSplitter and processing reports...")
    print("(This may involve reading multiple files and can take a moment)...")
    splitter = TextSplitter(
        # Default settings are often sensible:
        # chunk_size=1000 characters, chunk_overlap=200 characters.
        # These can be customized if needed, e.g.,
        # chunk_size=1000,
        # chunk_overlap=200,
    )

    try:
        # `split_all_reports` processes each JSON file in `all_report_dir`.
        # For each report, it also looks for a corresponding report in `serialized_tables_dir`
        # to extract serialized table data for separate chunking.
        splitter.split_all_reports(
            all_report_dir=input_merged_reports_dir,    # Path to merged reports
            output_dir=chunked_output_dir,              # Where to save chunked reports
            serialized_tables_dir=serialized_tables_input_dir # Path to reports with serialized tables
        )
        print("Text splitting process complete.")
        print(f"Chunked report should be available at: {chunked_output_path}")
    except Exception as e:
        print(f"Error during text splitting: {e}")
        # Potentially print more detailed traceback if in debug mode
        import traceback
        traceback.print_exc()
        return

    # --- 5. Load and Display Chunked Report Data ---
    print("\n--- Chunked Report Data ---")
    if not chunked_output_path.exists():
        print(f"Error: Chunked report file not found at {chunked_output_path}")
        print("The splitting process may have failed to produce an output.")
        if chunked_output_dir.exists():
            print(f"Contents of '{chunked_output_dir}': {list(chunked_output_dir.iterdir())}")
        return

    try:
        with open(chunked_output_path, 'r', encoding='utf-8') as f:
            chunked_data = json.load(f)

        # --- 5.1. Metainfo (should be preserved) ---
        print("\n  Metainfo (from chunked report):")
        if 'metainfo' in chunked_data and chunked_data['metainfo']:
            for key, value in chunked_data['metainfo'].items():
                print(f"    {key}: {value}")
        else:
            print("    No 'metainfo' section found.")

        # --- 5.2. Content Structure ---
        print("\n  Content Structure:")
        if 'content' in chunked_data:
            print("    `content` key found.")
            if 'pages' in chunked_data['content']:
                print(f"    `content['pages']` found (contains {len(chunked_data['content']['pages'])} pages - structure preserved).")
            else:
                print("    `content['pages']` NOT found.")

            if 'chunks' in chunked_data['content']:
                num_chunks = len(chunked_data['content']['chunks'])
                print(f"    `content['chunks']` found: Total {num_chunks} chunks generated.")

                # --- 5.3. Details of First Few Chunks ---
                print("\n  Details of First 2-3 Chunks:")
                for i, chunk in enumerate(chunked_data['content']['chunks'][:3]):
                    print(f"    --- Chunk {i+1} ---")
                    print(f"      ID: {chunk.get('id')}")
                    print(f"      Type: {chunk.get('type')} (e.g., 'content' or 'serialized_table')")
                    print(f"      Page: {chunk.get('page_number')}") # Page number of the source
                    print(f"      Length (tokens): {chunk.get('length_tokens')}") # Estimated token count
                    text_snippet = chunk.get('text', '')[:200] # First 200 chars
                    print(f"      Text Snippet: \"{text_snippet}...\"")
                if num_chunks > 3:
                    print("    ...")
            else:
                print("    `content['chunks']` NOT found. Splitting might have had an issue.")
        else:
            print("  No 'content' section found in the chunked report.")

    except json.JSONDecodeError:
        print(f"  Error: Could not decode the chunked JSON file at {chunked_output_path}.")
    except Exception as e:
        print(f"  An error occurred while loading or displaying the chunked JSON: {e}")
        import traceback
        traceback.print_exc()
    print("---------------------------")

    # --- 6. Cleanup (Optional) ---
    # To clean up the created directory:
    # try:
    #     shutil.rmtree(chunked_output_dir)
    #     print(f"\nSuccessfully removed chunked reports directory: {chunked_output_dir}")
    # except OSError as e:
    #     print(f"\nError removing chunked reports directory {chunked_output_dir}: {e.strerror}")
    print(f"\nDemo complete. Chunked report is in: {chunked_output_dir}")
    print("You can inspect the chunked JSON file there or manually delete the directory.")

if __name__ == "__main__":
    main()
