# study/demo_01_pdf_parsing.py

import sys
import os
import json
from pathlib import Path

# Add the src directory to the Python path to allow importing from src
# This is necessary for the script to find the `src.pdf_parsing` module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pdf_parsing import PDFParser # The primary class for parsing PDFs
from src.pipeline import Pipeline # Required for downloading Docling models

def main():
    """
    Demonstrates parsing a PDF file using PDFParser and exporting its content to JSON.
    PDFParser utilizes DoclingParseV2DocumentBackend (and underlying Docling models)
    for comprehensive document understanding, including text extraction, layout
    detection, structure recognition, and entity recognition.
    """
    print("Starting PDF parsing demo...")

    # --- 1. Ensure Docling models are downloaded ---
    # PDFParser relies on models provided by the Docling library.
    # This step ensures that these models are available.
    # It might take a while on the first run if models need to be downloaded.
    print("Checking and downloading Docling models if necessary...")
    try:
        Pipeline.download_docling_models()
        print("Docling models are ready.")
    except Exception as e:
        print(f"Error downloading or verifying Docling models: {e}")
        print("Please check your internet connection and model download path configuration.")
        return

    # --- 2. Define Input and Output Paths ---
    # The input PDF path is relative to the project root.
    sample_pdf_filename = "194000c9109c6fa628f1fed33b44ae4c2b8365f4.pdf"
    sample_pdf_path = Path(f"data/test_set/pdf_reports/{sample_pdf_filename}")

    # The output directory for the parsed JSON data.
    output_dir = Path("study/parsed_output")
    # Create the output directory if it doesn't exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    # The output JSON file path will be inside the output_dir.
    output_json_path = output_dir / f"{sample_pdf_filename}.json"

    print(f"Input PDF: {sample_pdf_path}")
    print(f"Output JSON will be saved to: {output_json_path}")

    # Check if the sample PDF file exists
    if not sample_pdf_path.exists():
        print(f"Error: Sample PDF file not found at {sample_pdf_path}")
        print("Please ensure the data is available in the 'data/test_set/pdf_reports/' directory.")
        # To help debug, show the absolute path being checked
        print(f"Absolute path checked: {sample_pdf_path.resolve()}")
        # List files in the expected directory if it exists
        if sample_pdf_path.parent.exists():
             print(f"Files in '{sample_pdf_path.parent}': {list(sample_pdf_path.parent.iterdir())[:5]}")
        else:
            print(f"Directory '{sample_pdf_path.parent}' does not exist.")
        return

    # --- 3. Initialize PDFParser ---
    # PDFParser is the main interface for parsing documents.
    # It internally uses DoclingParseV2DocumentBackend which orchestrates
    # various models (text, layout, structure, entities).
    # The `output_dir` specified here is where PDFParser will save its output,
    # including the final JSON representation of the parsed document.
    print(f"Initializing PDFParser with output directory: {output_dir}...")
    pdf_parser = PDFParser(output_dir=str(output_dir))

    # --- 4. Parse the PDF and Export to JSON ---
    # The `parse_and_export` method processes the PDF specified by `input_doc_paths`.
    # It handles the entire pipeline:
    #   - Converts PDF pages to images.
    #   - Runs text detection and OCR (if needed).
    #   - Performs layout analysis to identify elements like paragraphs, tables, figures.
    #   - Infers structural relationships between elements.
    #   - Recognizes named entities.
    # The result is saved as a JSON file in the `output_dir` of the PDFParser.
    print(f"Parsing PDF: {sample_pdf_path}...")
    try:
        # `parse_and_export` expects a list of Path objects.
        pdf_parser.parse_and_export(input_doc_paths=[sample_pdf_path])
        print("PDF parsing and export process initiated.")
    except Exception as e:
        print(f"Error during PDF parsing or export: {e}")
        print("This could be due to issues with the PDF file, model incompatibilities, or resource limits.")
        return

    # --- 5. Load and Print Output JSON ---
    # Verify that the output JSON file was created by PDFParser.
    if output_json_path.exists():
        print(f"\nSuccessfully parsed and exported. Output JSON available at: {output_json_path}")
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                parsed_data = json.load(f)

            print("\n--- Snippet of Parsed JSON Output ---")
            # Print metainfo if available
            if 'metainfo' in parsed_data:
                print("Metainfo:")
                for key, value in parsed_data['metainfo'].items():
                    print(f"  {key}: {value}")
            else:
                # If no metainfo, print a small part of the JSON string
                print("Printing the first 500 characters of the JSON output:")
                json_str_snippet = json.dumps(parsed_data, indent=2)[:500]
                print(json_str_snippet + "...")
            print("------------------------------------")

        except json.JSONDecodeError:
            print(f"Error: Could not decode the JSON file at {output_json_path}.")
            print("The file might be corrupted or not a valid JSON.")
        except Exception as e:
            print(f"An error occurred while reading or printing the JSON output: {e}")
    else:
        print(f"\nError: Output JSON file not found at {output_json_path}.")
        print("The parsing process may have failed to produce an output.")
        print(f"Please check logs or errors from the PDFParser execution.")
        print(f"Expected output file: {output_json_path.resolve()}")
        if output_dir.exists():
            print(f"Contents of output directory '{output_dir}': {list(output_dir.iterdir())}")
        else:
            print(f"Output directory '{output_dir}' does not exist.")


if __name__ == "__main__":
    main()