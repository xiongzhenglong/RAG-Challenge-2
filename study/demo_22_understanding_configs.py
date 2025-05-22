# study/demo_22_understanding_configs.py

import sys
import os
from pathlib import Path

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import (
    PipelineConfig,
    RunConfig,
    base_config,          # Example predefined RunConfig
    max_nst_o3m_config,   # Example predefined RunConfig
    # Add other predefined configs if you want to explore them
    # e.g., nst_config, no_reranking_config
)

# --- Purpose of this Demo ---
# This script explores the `PipelineConfig` and `RunConfig` classes from src/pipeline.py.
# These classes are crucial for controlling the behavior, data paths, and operational
# parameters of the main `Pipeline` used in the financial document processing project.
#
# - `PipelineConfig`: Primarily manages directory paths for inputs and outputs at
#   various stages of the pipeline. Its paths can change based on whether
#   serialized table data is being used (affecting directory naming conventions).
# - `RunConfig`: Manages boolean flags and parameters that control how the pipeline
#   executes, such as whether to use table serialization, LLM reranking,
#   parallelism settings, etc. Predefined instances of `RunConfig` offer
#   ready-to-use configurations for different scenarios.

def print_config_attributes(config_obj, title="Configuration Attributes"):
    """Helper function to print attributes of a config object."""
    print(f"\n--- {title} ({config_obj.__class__.__name__}) ---")
    if not config_obj:
        print("  Object is None.")
        return
        
    # Basic explanation mapping for some attributes
    # This can be expanded for more clarity
    explanations = {
        # PipelineConfig attributes
        "root_path": "Root directory for all pipeline data.",
        "pdf_reports_dir": "Directory for input PDF reports.",
        "parsed_reports_dirname": "Base name for directory storing parsed JSON reports.",
        "parsed_reports_path": "Full path to directory storing parsed JSON reports.",
        "merged_reports_dirname": "Base name for directory storing merged/simplified JSON reports.",
        "merged_reports_path": "Full path to directory storing merged/simplified JSON reports.",
        "reports_markdown_dirname": "Base name for directory storing exported Markdown reports.",
        "reports_markdown_path": "Full path to directory storing exported Markdown reports.",
        "chunked_reports_dirname": "Base name for directory storing chunked JSON reports.",
        "chunked_reports_path": "Full path to directory storing chunked JSON reports.",
        "databases_path": "Directory for storing generated databases (FAISS, BM25 indices).",
        "faiss_indices_dirname": "Subdirectory name for FAISS indices within databases_path.",
        "faiss_indices_path": "Full path to FAISS indices directory.",
        "bm25_indices_dirname": "Subdirectory name for BM25 indices within databases_path.",
        "bm25_indices_path": "Full path to BM25 indices directory.",
        "debug_data_path": "Directory for storing intermediate debug data.",
        "subset_path": "Full path to the subset CSV file.",
        "questions_path": "Full path to the questions JSON file.",
        "answers_path": "Full path for saving the final answers JSON file.",
        "serialized": "Boolean, if True, paths reflect usage of serialized table data.",
        "config_suffix": "Suffix appended to output directory names (from RunConfig).",

        # RunConfig attributes
        "use_serialized_tables": "Boolean - Whether to use serialized table data during parsing/processing.",
        "parent_document_retrieval": "Boolean - Whether to use parent document retrieval strategy.",
        "llm_reranking": "Boolean - Whether to use LLM-based reranking of retrieved chunks.",
        "config_suffix": "String - Suffix to append to output directory names for this run.",
        "parallel_requests": "Integer - Number of parallel requests for API calls.",
        "top_n_retrieval": "Integer - Number of top chunks to retrieve initially.",
        "llm_reranking_sample_size": "Integer - Number of initially retrieved chunks to pass to LLM for reranking.",
        "subset_name": "String - Name of the subset CSV file.",
        "questions_file_name": "String - Name of the questions JSON file.",
        "pdf_reports_dir_name": "String - Name of the directory containing PDF reports.",
        "submission_file": "Boolean - Whether to generate a submission file (e.g., for a competition)."
    }

    for attr, value in vars(config_obj).items():
        explanation = explanations.get(attr, "No specific explanation available.")
        print(f"  self.{attr}: {value!r}  # {explanation}")
    print("-" * 40)

def main():
    """
    Demonstrates PipelineConfig and RunConfig.
    """
    print("--- Understanding PipelineConfig and RunConfig ---")

    # --- 1. Demonstrate PipelineConfig ---
    print("\n" + "="*60)
    print("=== PipelineConfig Demonstration ===")
    print("="*60)
    print("`PipelineConfig` manages the directory structure for the pipeline's data.")
    print("Its paths are dynamically constructed based on the `root_path` and a `serialized` flag.")

    sample_root = Path("./study/pipeline_config_demo_root") # Using a subdirectory in study for demo

    # Example 1.1: PipelineConfig with serialized=False
    print("\nExample 1.1: PipelineConfig with `serialized=False` (default for raw parsing output)")
    pipeline_conf_default = PipelineConfig(root_path=sample_root, serialized=False, config_suffix="_default_run")
    print_config_attributes(pipeline_conf_default, "PipelineConfig (serialized=False)")
    print(f"  Note: `parsed_reports_path` is '{pipeline_conf_default.parsed_reports_path.name}'")
    print(f"  And `databases_path` is '{pipeline_conf_default.databases_path.name}'")


    # Example 1.2: PipelineConfig with serialized=True
    print("\nExample 1.2: PipelineConfig with `serialized=True`")
    print("This might be used for stages after table serialization, where outputs are stored differently.")
    pipeline_conf_serialized = PipelineConfig(root_path=sample_root, serialized=True, config_suffix="_serialized_run")
    print_config_attributes(pipeline_conf_serialized, "PipelineConfig (serialized=True)")
    print(f"  Note: `parsed_reports_path` is now '{pipeline_conf_serialized.parsed_reports_path.name}' (contains '_st')")
    print(f"  And `databases_path` is now '{pipeline_conf_serialized.databases_path.name}' (contains '_st')")
    print("  The '_st' (serialized tables) suffix is added to relevant directory names.")

    # --- 2. Demonstrate RunConfig ---
    print("\n" + "="*60)
    print("=== RunConfig Demonstration ===")
    print("="*60)
    print("`RunConfig` holds parameters that control the pipeline's execution logic.")

    # Example 2.1: Default RunConfig
    print("\nExample 2.1: Default RunConfig instance")
    default_run_config = RunConfig()
    print_config_attributes(default_run_config, "Default RunConfig")

    # Example 2.2: Predefined RunConfig - base_config
    print("\nExample 2.2: Predefined `base_config`")
    print("This configuration represents a standard, balanced setup.")
    print_config_attributes(base_config, "Predefined 'base_config'")

    # Example 2.3: Predefined RunConfig - max_nst_o3m_config
    print("\nExample 2.3: Predefined `max_nst_o3m_config`")
    print("This configuration is likely optimized for maximum performance or specific features,")
    print("e.g., 'nst' might mean 'no serialized tables', 'o3m' could refer to a model or setting.")
    print_config_attributes(max_nst_o3m_config, "Predefined 'max_nst_o3m_config'")


    # --- 3. Illustrate Interaction between RunConfig and PipelineConfig ---
    print("\n" + "="*60)
    print("=== Interaction: RunConfig influencing PipelineConfig via Pipeline ===")
    print("="*60)
    print("When the main `Pipeline` class is initialized, it takes a `RunConfig` instance.")
    print("The `Pipeline` class then uses settings from this `RunConfig` to instantiate its own `PipelineConfig`.")
    print("\nSpecifically, `PipelineConfig`'s `serialized` flag and `config_suffix` are often derived from `RunConfig`:")
    print("  - `PipelineConfig(serialized = run_config.use_serialized_tables, ...)`")
    print("  - `PipelineConfig(config_suffix = run_config.config_suffix, ...)`")
    print("\nLet's simulate this with `base_config` which has `use_serialized_tables=True`:")
    
    simulated_pipeline_config_for_base = PipelineConfig(
        root_path=sample_root,
        serialized=base_config.use_serialized_tables, # From RunConfig
        config_suffix=base_config.config_suffix      # From RunConfig
    )
    print_config_attributes(simulated_pipeline_config_for_base, 
                            f"PipelineConfig if Pipeline used 'base_config' (serialized={base_config.use_serialized_tables})")
    print(f"  Resulting `parsed_reports_path` name: '{simulated_pipeline_config_for_base.parsed_reports_path.name}'")
    print(f"  Resulting `databases_path` name: '{simulated_pipeline_config_for_base.databases_path.name}'")
    print("  Notice the '_st' and '_base' suffixes, derived from `base_config`'s settings.")

    print("\nThis mechanism allows a single `Pipeline` class to adapt its data paths and behavior")
    print("based on the provided `RunConfig`, making it flexible for different experimental setups.")

    print("\n\nConfiguration exploration complete.")
    print(f"Note: A dummy directory '{sample_root.resolve()}' might have been implicitly referenced "
          f"but not necessarily created unless paths were accessed in a way that triggers creation.")

if __name__ == "__main__":
    main()
