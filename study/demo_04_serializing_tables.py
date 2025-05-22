# study/demo_04_serializing_tables.py

import json
import os
import shutil
from pathlib import Path
import sys

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tables_serialization import TableSerializer

def main():
    """
    Demonstrates table serialization using `TableSerializer`.
    This process enriches table data with contextual information,
    making it more suitable for Retrieval Augmented Generation (RAG) systems.
    """
    print("Starting table serialization demo...")

    # --- 1. Define Paths ---
    original_json_filename = "194000c9109c6fa628f1fed33b44ae4c2b8365f4.json" # Example file
    original_json_path = Path("study/parsed_output") / original_json_filename

    # Temporary directory and file for serialization to avoid modifying the original
    copied_json_dir = Path("study/temp_serialization_data")
    copied_json_filename = "report_for_serialization.json"
    copied_json_path = copied_json_dir / copied_json_filename

    print(f"Original parsed JSON: {original_json_path}")
    print(f"Temporary file for serialization: {copied_json_path}")

    # --- 2. Prepare for Serialization (Copy File) ---
    if not original_json_path.exists():
        print(f"Error: Original JSON file not found at {original_json_path}")
        print("Please ensure 'demo_01_pdf_parsing.py' has been run successfully.")
        return

    # Create the temporary directory if it doesn't exist
    copied_json_dir.mkdir(parents=True, exist_ok=True)

    # Copy the original JSON to the temporary location (overwrite if exists)
    shutil.copy(original_json_path, copied_json_path)
    print(f"Copied original JSON to temporary location: {copied_json_path}")

    # --- 3. Understanding Table Serialization ---
    # Table serialization, in this context, refers to the process of transforming
    # structured table data (like rows and columns, often in HTML or Markdown)
    # into a more descriptive and context-aware format.
    #
    # Why is this useful for RAG?
    # - LLMs work best with natural language. Raw table structures (e.g., HTML tags,
    #   Markdown pipes) can be noisy and difficult for LLMs to interpret directly.
    # - Context is key: Simply having the cell values isn't enough. LLMs need to
    #   understand what the table is about, what its main entities are, and how
    #   different pieces of information relate to each other.
    # - Searchability: Serialized text blocks are easier to index and search for
    #   relevant information when a user asks a question.
    #
    # This implementation (`TableSerializer`) uses an LLM (e.g., OpenAI's GPT) to:
    #   1. Identify the main subject or entities discussed in the table.
    #   2. Determine which table headers are most relevant to these subjects.
    #   3. Generate "information blocks": These are natural language sentences or
    #      paragraphs that summarize key information from the table, often focusing
    #      on a specific entity and its related data points from relevant columns.
    # The goal is to create self-contained, context-rich textual representations
    # of the table's core information.

    # --- 4. Load Original Table Data (Before Serialization) ---
    print("\n--- Original Table Data (Before Serialization) ---")
    try:
        with open(copied_json_path, 'r', encoding='utf-8') as f:
            data_before_serialization = json.load(f)

        if 'tables' in data_before_serialization and data_before_serialization['tables']:
            first_table_before = data_before_serialization['tables'][0]
            print(f"  Table ID: {first_table_before.get('table_id', 'N/A')}")
            print(f"  Page: {first_table_before.get('page', 'N/A')}")
            
            # Displaying HTML as it's often a rich representation available
            html_repr_before = first_table_before.get('html', 'N/A')
            if html_repr_before != 'N/A':
                print(f"  HTML Representation (Snippet):\n{html_repr_before[:500]}...")
            else:
                # Fallback to Markdown if HTML is not present
                markdown_repr_before = first_table_before.get('markdown', 'No Markdown representation found.')
                print(f"  Markdown Representation (Snippet):\n{markdown_repr_before[:500]}...")
        else:
            print("  No tables found in the original JSON data.")
            # If no tables, serialization won't do much, so we can stop.
            # Clean up and exit.
            # shutil.rmtree(copied_json_dir)
            # print(f"\nCleaned up temporary directory: {copied_json_dir}")
            return
    except Exception as e:
        print(f"  Error loading or displaying original table data: {e}")
        # Clean up and exit if we can't load the data.
        # shutil.rmtree(copied_json_dir)
        # print(f"\nCleaned up temporary directory: {copied_json_dir}")
        return
    print("----------------------------------------------------")

    # --- 5. Perform Serialization ---
    # The TableSerializer modifies the JSON file in place.
    # It iterates through each table, generates serialized content using an LLM,
    # and adds it under a "serialized" key within each table object.
    print("\nInitializing TableSerializer and processing the file...")
    print("(This may take some time as it involves LLM calls for each table)...")
    
    # Make sure OPENAI_API_KEY is set in your environment for the serializer to work.
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("TableSerializer requires an OpenAI API key to function.")
        print("Please set it and try again.")
        # Clean up and exit.
        # shutil.rmtree(copied_json_dir)
        # print(f"\nCleaned up temporary directory: {copied_json_dir}")
        return

    try:
        serializer = TableSerializer() # Uses OPENAI_API_KEY from environment
        serializer.process_file(copied_json_path) # Modifies the file in-place
        print("Table serialization process complete.")
        print(f"The file {copied_json_path} has been updated with serialized table data.")
    except Exception as e:
        print(f"Error during table serialization: {e}")
        print("This could be due to API issues, configuration problems, or issues with the table data itself.")
        # Clean up and exit.
        # shutil.rmtree(copied_json_dir)
        # print(f"\nCleaned up temporary directory: {copied_json_dir}")
        return

    # --- 6. Load and Display Serialized Table Data ---
    print("\n--- Serialized Table Data (After Serialization) ---")
    try:
        with open(copied_json_path, 'r', encoding='utf-8') as f:
            data_after_serialization = json.load(f)

        if 'tables' in data_after_serialization and data_after_serialization['tables']:
            # Assuming we are interested in the same first table
            first_table_after = data_after_serialization['tables'][0]
            print(f"  Inspecting Table ID: {first_table_after.get('table_id', 'N/A')}")

            if 'serialized' in first_table_after and first_table_after['serialized']:
                serialized_content = first_table_after['serialized']

                # `subject_core_entities_list`: Main entities the table is about.
                print("\n  1. Subject Core Entities List:")
                print("     (Identified by LLM as the primary subjects of the table)")
                entities = serialized_content.get('subject_core_entities_list', [])
                if entities:
                    for entity in entities:
                        print(f"       - {entity}")
                else:
                    print("       No core entities identified or list is empty.")

                # `relevant_headers_list`: Headers most relevant to the core entities.
                print("\n  2. Relevant Headers List:")
                print("     (Headers LLM deemed most important for understanding the entities)")
                headers = serialized_content.get('relevant_headers_list', [])
                if headers:
                    for header in headers:
                        print(f"       - {header}")
                else:
                    print("       No relevant headers identified or list is empty.")

                # `information_blocks`: LLM-generated natural language summaries.
                print("\n  3. Information Blocks (Sample):")
                print("     (LLM-generated sentences combining entities with their relevant data from the table)")
                blocks = serialized_content.get('information_blocks', [])
                if blocks:
                    for i, block_item in enumerate(blocks[:2]): # Show first two blocks
                        print(f"     Block {i+1}:")
                        print(f"       Subject Core Entity: {block_item.get('subject_core_entity', 'N/A')}")
                        print(f"       Information Block Text: \"{block_item.get('information_block', 'N/A')}\"")
                else:
                    print("       No information blocks generated or list is empty.")
            else:
                print("  'serialized' key not found in the table object or is empty.")
                print("  This might indicate an issue during the serialization process for this table.")
        else:
            print("  No tables found in the JSON data after serialization (unexpected).")

    except Exception as e:
        print(f"  Error loading or displaying serialized table data: {e}")
    print("-----------------------------------------------------")

    # --- 7. Cleanup (Optional) ---
    # Uncomment the following lines to remove the temporary directory after the demo.
    # try:
    #     shutil.rmtree(copied_json_dir)
    #     print(f"\nSuccessfully removed temporary directory: {copied_json_dir}")
    # except OSError as e:
    #     print(f"\nError removing temporary directory {copied_json_dir}: {e.strerror}")
    print(f"\nDemo complete. Temporary data is in: {copied_json_dir}")
    print("You can inspect the modified JSON file there or manually delete the directory.")


if __name__ == "__main__":
    main()
