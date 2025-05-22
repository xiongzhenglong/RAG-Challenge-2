# study/demo_05_merging_reports.py

import json
import os
import shutil
from pathlib import Path
import sys

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parsed_reports_merging import PageTextPreparation

def main():
    """
    Demonstrates merging and simplifying parsed report JSON using PageTextPreparation.
    This process consolidates page content into a single text string per page,
    optionally incorporating serialized table data.
    """
    print("Starting report merging demo...")

    # --- 1. Define Paths ---
    # Input is the output of demo_04 (which includes serialized tables)
    input_report_dir = Path("study/temp_serialization_data/")
    input_report_filename = "report_for_serialization.json" # File processed by demo_04
    input_report_full_path = input_report_dir / input_report_filename

    # Output directory for the merged and simplified report
    merged_output_dir = Path("study/merged_reports_output/")
    # The PageTextPreparation process will save the output file with the same name
    # as the input file, but in the specified output_dir.
    merged_output_path = merged_output_dir / input_report_filename

    print(f"Input report (from demo_04): {input_report_full_path}")
    print(f"Merged report output directory: {merged_output_dir}")

    # --- 2. Prepare Input Data ---
    if not input_report_full_path.exists():
        print(f"Error: Input report file not found at {input_report_full_path}")
        print("Please ensure 'demo_04_serializing_tables.py' has been run successfully,")
        print("as its output is used as input for this demo.")
        return

    # Create the merged output directory if it doesn't exist
    merged_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {merged_output_dir}")

    # --- 3. Understanding Report Merging with PageTextPreparation ---
    # `PageTextPreparation` is designed to simplify the complex JSON output
    # from `PDFParser` (which might have already been augmented by `TableSerializer`).
    # Its main goal is to produce a JSON where each page's content is represented
    # as a single, continuous string of text. This is highly beneficial for:
    #   - RAG systems: Simpler text chunks are easier to embed and retrieve.
    #   - Downstream NLP tasks: Many NLP models prefer plain text input.
    #
    # Key actions performed by `PageTextPreparation`:
    #   - Consolidates Content: It iterates through the structured page content
    #     (like paragraphs, headers, lists) and joins their text.
    #   - Applies Formatting Rules: It can apply rules to ensure consistent spacing,
    #     remove redundant newlines, etc.
    #   - Incorporates Serialized Tables (Optional):
    #     - If `use_serialized_tables=True`, it looks for the "serialized" data
    #       within each table object (as produced by `TableSerializer`).
    #     - If `serialized_tables_instead_of_markdown=True` and serialized data exists,
    #       it will use the `information_blocks` from the serialized table data
    #       instead of the table's Markdown or HTML representation. This provides
    #       more natural language context for tables. If serialized data is not found
    #       or `use_serialized_tables` is False, it falls back to Markdown/HTML.
    # The output JSON has a simpler structure, especially under `content['pages']`,
    # where each page object will have a direct 'text' key holding the consolidated string.

    # --- 4. Perform Merging ---
    print("\nInitializing PageTextPreparation and processing the report...")
    # We use `use_serialized_tables=True` and `serialized_tables_instead_of_markdown=True`
    # to demonstrate the inclusion of the rich, LLM-generated table summaries.
    preparator = PageTextPreparation(
        use_serialized_tables=True,
        serialized_tables_instead_of_markdown=True
    )

    try:
        # `process_reports` can handle multiple files in `reports_dir`.
        # For this demo, `input_report_dir` contains one file.
        preparator.process_reports(
            reports_dir=input_report_dir,
            output_dir=merged_output_dir
        )
        print("Report merging process complete.")
        print(f"Merged report should be available at: {merged_output_path}")
    except Exception as e:
        print(f"Error during report merging: {e}")
        return

    # --- 5. Load and Display Merged Report Data ---
    print("\n--- Merged Report Data ---")
    if not merged_output_path.exists():
        print(f"Error: Merged report file not found at {merged_output_path}")
        print("The merging process may have failed to produce an output.")
        # List contents of merged_output_dir to help debug
        if merged_output_dir.exists():
            print(f"Contents of '{merged_output_dir}': {list(merged_output_dir.iterdir())}")
        return

    try:
        with open(merged_output_path, 'r', encoding='utf-8') as f:
            merged_data = json.load(f)

        # --- 5.1. Metainfo (should be preserved) ---
        print("\n  Metainfo (from merged report):")
        if 'metainfo' in merged_data and merged_data['metainfo']:
            for key, value in merged_data['metainfo'].items():
                print(f"    {key}: {value}")
        else:
            print("    No 'metainfo' section found.")

        # --- 5.2. Content of the First Page (Simplified) ---
        print("\n  Content of First Page (from merged report - first 1000 chars):")
        if 'content' in merged_data and 'pages' in merged_data['content'] and merged_data['content']['pages']:
            first_page_merged = merged_data['content']['pages'][0]
            page_number = first_page_merged.get('page_number', 'N/A') # Key is 'page_number' here
            page_text = first_page_merged.get('text', '')

            print(f"    Page Number: {page_number}")
            print(f"    Consolidated Page Text (Snippet):\n\"{page_text[:1000]}...\"")

            # --- Comparison Note ---
            print("\n    --- Structural Comparison ---")
            print("    The original JSON (e.g., from demo_01 or demo_04) has a complex page structure:")
            print("    `content[0]['content']` would be a list of blocks (paragraphs, headers),")
            print("    each with its own text and type. Tables would be separate objects.")
            print("\n    In this merged report:")
            print("    `content['pages'][0]['text']` directly contains the FULL text of the page,")
            print("    with elements like paragraphs and (optionally serialized) tables integrated")
            print("    into this single string. This is much simpler for direct use in RAG.")
            print("    Serialized table 'information_blocks' should be part of this text if they were processed.")
            print("    -----------------------------")

        else:
            print("    No page content found in the expected simplified structure.")
            print("    Merged data structure:", json.dumps(merged_data.get('content', {}), indent=2)[:500])


    except json.JSONDecodeError:
        print(f"  Error: Could not decode the merged JSON file at {merged_output_path}.")
    except Exception as e:
        print(f"  An error occurred while loading or displaying the merged JSON: {e}")
    print("--------------------------")

    # --- 6. Cleanup (Optional) ---
    # To clean up the created directory:
    # try:
    #     shutil.rmtree(merged_output_dir)
    #     print(f"\nSuccessfully removed merged reports directory: {merged_output_dir}")
    # except OSError as e:
    #     print(f"\nError removing merged reports directory {merged_output_dir}: {e.strerror}")
    print(f"\nDemo complete. Merged report is in: {merged_output_dir}")
    print("You can inspect the merged JSON file there or manually delete the directory.")

if __name__ == "__main__":
    main()
