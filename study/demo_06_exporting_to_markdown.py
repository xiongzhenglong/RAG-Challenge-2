# study/demo_06_exporting_to_markdown.py

import os
from pathlib import Path
import sys

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parsed_reports_merging import PageTextPreparation

def main():
    """
    Demonstrates exporting the merged and simplified JSON report
    (output of demo_05_merging_reports.py) to a Markdown file.
    """
    print("Starting Markdown export demo...")

    # --- 1. Define Paths ---
    # Input is the directory containing merged JSON reports (output of demo_05)
    input_merged_reports_dir = Path("study/merged_reports_output/")
    # We assume the same filename was processed through the demos
    input_report_filename = "report_for_serialization.json"
    expected_input_json_path = input_merged_reports_dir / input_report_filename

    # Output directory for the exported Markdown files
    markdown_output_dir = Path("study/markdown_export_output/")
    # The export_to_markdown method will create a .md file with the same base name
    output_markdown_filename = input_report_filename.replace(".json", ".md")
    output_markdown_path = markdown_output_dir / output_markdown_filename

    print(f"Input merged JSON directory: {input_merged_reports_dir}")
    print(f"Expected input JSON file: {expected_input_json_path}")
    print(f"Markdown output directory: {markdown_output_dir}")
    print(f"Expected output Markdown file: {output_markdown_path}")

    # --- 2. Prepare Input Data ---
    if not expected_input_json_path.exists():
        print(f"Error: Expected input merged JSON file not found at {expected_input_json_path}")
        print("Please ensure 'demo_05_merging_reports.py' has been run successfully,")
        print("as its output directory is used as input for this demo.")
        return

    # Create the Markdown output directory if it doesn't exist
    markdown_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured Markdown output directory exists: {markdown_output_dir}")

    # --- 3. Understanding Markdown Export ---
    # `PageTextPreparation.export_to_markdown()` takes the simplified JSON reports
    # (like those produced by `PageTextPreparation.process_reports()` in demo_05)
    # and converts them into human-readable Markdown documents.
    #
    # Each JSON report file in the input directory will result in a corresponding .md file.
    # The Markdown file typically contains:
    #   - Document Metainfo: Often included at the beginning (e.g., filename, SHA).
    #   - Page Content: The consolidated text from each page (`page['text']`) is written out.
    #     Page breaks or separators (like "--- Page X ---") might be included.
    #
    # This Markdown export is useful for:
    #   - Easy Review: Quickly read the textual content of the parsed PDF.
    #   - Version Control: Store a text-based representation of the document.
    #   - Certain types of full-text processing or indexing where Markdown is a preferred input.
    #   - Basic sharing or reporting when a simple text format is needed.
    #
    # The `PageTextPreparation` instance is initialized with the same settings
    # (`use_serialized_tables`, `serialized_tables_instead_of_markdown`) as in demo_05
    # for consistency. While these settings primarily affect the `process_reports` method
    # (which generates the input for this script), `export_to_markdown` might have
    # internal logic that expects or benefits from knowing how its input was generated,
    # particularly if it needs to interpret structure that was influenced by these settings.

    # --- 4. Perform Markdown Export ---
    print("\nInitializing PageTextPreparation and exporting to Markdown...")
    # Using the same settings as demo_05 for consistency, as the input to this
    # script is the output of demo_05.
    preparator = PageTextPreparation(
        use_serialized_tables=True,
        serialized_tables_instead_of_markdown=True
    )

    try:
        # The `export_to_markdown` method processes all .json files in `reports_dir`
        # and saves them as .md files in `output_dir`.
        preparator.export_to_markdown(
            reports_dir=input_merged_reports_dir, # Must be Path object
            output_dir=markdown_output_dir      # Must be Path object
        )
        print("Markdown export process complete.")
        print(f"Markdown file should be available at: {output_markdown_path}")
    except Exception as e:
        print(f"Error during Markdown export: {e}")
        return

    # --- 5. Show Snippet of Markdown (Optional) ---
    print("\n--- Snippet of Exported Markdown ---")
    if output_markdown_path.exists():
        try:
            with open(output_markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            print(f"Successfully read Markdown file: {output_markdown_path}")
            print("First 1500 characters of the Markdown content:\n")
            print(markdown_content[:1500])
            if len(markdown_content) > 1500:
                print("\n[... content truncated ...]")
        except Exception as e:
            print(f"Error reading or displaying the Markdown file: {e}")
    else:
        print(f"Markdown file not found at {output_markdown_path}.")
        print("The export process may have failed to produce an output.")
        # List contents of markdown_output_dir to help debug
        if markdown_output_dir.exists():
            print(f"Contents of '{markdown_output_dir}': {list(markdown_output_dir.iterdir())}")

    print("------------------------------------")

    # --- 6. Cleanup (Optional) ---
    # To clean up the created directory:
    # import shutil
    # try:
    #     shutil.rmtree(markdown_output_dir)
    #     print(f"\nSuccessfully removed Markdown export directory: {markdown_output_dir}")
    # except OSError as e:
    #     print(f"\nError removing Markdown export directory {markdown_output_dir}: {e.strerror}")
    print(f"\nDemo complete. Exported Markdown is in: {markdown_output_dir}")
    print("You can inspect the .md file there or manually delete the directory.")

if __name__ == "__main__":
    main()
