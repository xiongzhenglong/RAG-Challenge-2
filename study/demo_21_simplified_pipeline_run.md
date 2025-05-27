# RAG 超融合：`demo_21_simplified_pipeline_run.py` 之简化版全流程运行

大家好！欢迎来到我们 PDF 文档智能处理与检索增强生成（RAG）系列教程的终极篇章——`demo_21_simplified_pipeline_run.py`！在过去的20个 Demo 中，我们如同探险家一般，一步一个脚印地探索了 RAG 系统的各个神秘角落：从最初的 PDF 文档解析，到形形色色的数据处理与索引构建，再到多样化的信息检索技术，以及如何与大型语言模型（LLM）巧妙互动以获取普通答案、结构化信息，乃至处理复杂的比较型问题和优化检索结果。

现在，是时候将所有这些珍珠串联起来，展现一幅完整的 RAG 画卷了。本篇教程将通过 `study/demo_21_simplified_pipeline_run.py` 脚本，使用 `src.pipeline.Pipeline` 类来驱动一个简化但功能完备的端到端 RAG 流程。我们将从一份 PDF 文档开始，经历解析、处理、索引，并最终对预设的问题生成答案，全程由 `Pipeline` 类根据配置自动调度。

## 脚本目标

- 演示如何使用 `Pipeline` 类和 `RunConfig` 配置对象来执行一个从 PDF 到答案的完整 RAG 工作流。
- 理解 `Pipeline` 类如何编排和集成项目中之前介绍的各个核心组件（解析、表格序列化、后处理、索引、问答等）。
- 展示如何为演示目的设置一个独立的、小型的、自包含的数据集。
- 观察整个流程中各个阶段的输入、输出以及最终生成的答案。

## `Pipeline` 类与 `RunConfig` 配置

-   **`Pipeline` 类 (`src.pipeline.Pipeline`)**:
    -   可以将其视为整个 RAG 工作流的总指挥或编排者。
    -   它封装了执行 RAG 各个阶段（如PDF解析、表格序列化、文本后处理、索引构建、问题处理等）的逻辑。
    -   通过调用 `Pipeline` 实例的不同方法，可以按顺序触发这些阶段的执行。
-   **`RunConfig` 类 (`src.pipeline.RunConfig`)**:
    -   这是一个配置类，用于向 `Pipeline` 实例传递运行时的各种参数和开关。
    -   通过 `RunConfig`，我们可以定制化 `Pipeline` 的行为，例如：
        -   是否使用表格序列化 (`use_serialized_tables`)。
        -   是否启用 LLM 重排序 (`llm_reranking`)。
        -   输出目录和文件的后缀 (`config_suffix`)，以区分不同的运行配置。
        -   API 调用的并行度 (`parallel_requests`)。
        -   检索和重排序时返回的条目数量 (`top_n_retrieval`, `llm_reranking_sample_size`)。
        -   等等。

这种设计使得 RAG 流程既能被高度自动化地执行，又能通过配置进行灵活调整。

## 为演示特别设置数据：`setup_pipeline_demo_data` 函数

为了让本 Demo 能够独立运行，不依赖于庞大的完整数据集或复杂的外部设置，脚本首先通过 `setup_pipeline_demo_data` 函数创建了一个小型的、隔离的演示数据集。

**`setup_pipeline_demo_data` 的主要工作：**

1.  **创建根目录**: 在 `study/` 下创建一个名为 `pipeline_demo_data` 的根目录。如果已存在，则先清空。
2.  **准备 PDF**:
    -   在 `pipeline_demo_data` 下创建 `pdf_reports` 子目录。
    -   从项目的主数据集路径（`data/test_set/pdf_reports/`）复制一个指定的 PDF 文件（例如 `194000c9109c6fa628f1fed33b44ae4c2b8365f4.pdf`）到这个 `pdf_reports` 目录，并将其重命名为 `democorp_pipeline_report.pdf`。
3.  **创建 `subset_demo.csv`**:
    -   在 `pipeline_demo_data` 根目录下创建一个 `subset_demo.csv` 文件。
    -   这个 CSV 文件模拟了主项目中用于指定处理哪些文档的 `subset.csv`。它包含一行数据，将我们复制的 PDF 文件（通过其主干名 `democorp_pipeline_report` 作为 `sha1` 字段）与一个虚构的公司名 "DemoCorp Pipeline" 等元数据关联起来。
4.  **创建 `questions_demo.json`**:
    -   在 `pipeline_demo_data` 根目录下创建一个 `questions_demo.json` 文件。
    -   这个 JSON 文件包含一到两个预设的问题，这些问题将针对 "DemoCorp Pipeline"（即我们准备的 PDF 内容）提出。问题的 `kind`（如 "number", "name"）指定了期望的答案类型或模式。

通过这个函数，我们为 `Pipeline` 的运行准备好了一个完整的、自洽的微型“项目环境”。

## 前提条件

1.  **`OPENAI_API_KEY` 环境变量**: 因为整个流程涉及到多次与 OpenAI API 的交互（表格序列化、嵌入生成、重排序、答案生成等）。
2.  **基础 PDF 文件**: `data/test_set/pdf_reports/194000c9109c6fa628f1fed33b44ae4c2b8365f4.pdf` 必须存在，作为 `setup_pipeline_demo_data` 的源文件。

## Python 脚本 `study/demo_21_simplified_pipeline_run.py`

让我们完整地看一下这个脚本的代码：
```python
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
```

## 脚本代码详解

### 1. 导入模块
```python
# ... (标准导入) ...
import pandas as pd # 用于创建演示用的 subset DataFrame
from src.pipeline import Pipeline, RunConfig # 核心：Pipeline 编排器和运行配置
```
- `pandas`: 用于方便地创建和保存 `subset_demo.csv` 文件。
- `Pipeline` 和 `RunConfig`: 这是本 Demo 的核心，用于驱动整个 RAG 流程。

### 2. `setup_pipeline_demo_data` 函数
此函数负责创建一套独立的、小型的演示数据集。
- **创建目录结构**: 在 `study/pipeline_demo_data` 下创建包括 `pdf_reports` 在内的子目录。如果目录已存在，则先清空。
- **复制 PDF**: 将 `data/test_set/pdf_reports/` 下的一个示例 PDF 复制到演示的 `pdf_reports` 目录，并重命名为 `democorp_pipeline_report.pdf`。
- **创建 `subset_demo.csv`**:
    - 使用 `pandas` 创建一个 DataFrame，其中包含一行数据。
    - `sha1`: 设置为 `democorp_pipeline_report` (PDF 文件的主干名)，这是文档的唯一ID。
    - `company_name`: 设置为 "DemoCorp Pipeline"。
    - 其他字段如 `document_type`, `period_end_on` 等也填入示例值。
    - 将 DataFrame 保存为 `subset_demo.csv`。这个文件告诉 `Pipeline` 要处理哪些文档。
- **创建 `questions_demo.json`**:
    - 创建一个包含两个示例问题的 JSON 文件。
    - 每个问题对象包含 `id`, `text` (问题文本，其中公司名动态替换为 "DemoCorp Pipeline") 和 `kind` (期望的答案类型/模式，如 "number", "name")。这个文件将作为 `Pipeline` 处理问题的输入。
- **返回**: 成功则返回演示数据根目录 `demo_root_path`。

### 3. `find_answers_file` 函数
这是一个辅助函数，用于查找 `Pipeline` 生成的答案文件。因为 `Pipeline` 内部的 `_get_next_available_filename` 可能会在文件名后添加数字后缀（如 `_01`, `_02`）以避免覆盖，所以这个函数会尝试查找这些可能的变体。

### 4. `main()` 函数

#### 4.1. 检查 API 密钥和设置演示数据
```python
    # ... (API Key 检查) ...
    demo_data_root = setup_pipeline_demo_data()
    if not demo_data_root:
        # ... (数据准备失败则退出) ...
        return
```
- 首先检查 `OPENAI_API_KEY`。
- 然后调用 `setup_pipeline_demo_data()` 创建本 Demo 所需的独立数据集。

#### 4.2. 配置和初始化 `Pipeline`
```python
    run_config = RunConfig(
        use_serialized_tables=True,
        parent_document_retrieval=False,
        llm_reranking=True,
        config_suffix="_demo_run", # 所有输出子目录和文件的后缀
        parallel_requests=1,
        top_n_retrieval=3,
        llm_reranking_sample_size=3,
        submission_file=False
    )
    # ...
    pipeline = Pipeline(
        root_path=demo_data_root,             # 指向我们刚创建的演示数据根目录
        subset_name="subset_demo.csv",        # 使用演示的 subset 文件
        questions_file_name="questions_demo.json", # 使用演示的问题文件
        pdf_reports_dir_name="pdf_reports",   # PDF 存放的子目录名
        run_config=run_config                 # 应用上述配置
    )
```
- **创建 `RunConfig`**:
    - `use_serialized_tables=True`: 启用表格序列化步骤（`demo_04` 相关）。
    - `parent_document_retrieval=False`: 为简化演示，不使用父子块检索。
    - `llm_reranking=True`: 启用 LLM 重排序步骤（`demo_16` 相关）。
    - `config_suffix="_demo_run"`: 非常重要！这个后缀会被附加到所有由 `Pipeline` 生成的中间数据目录名和输出文件名中（例如 `parsed_reports_json_demo_run`, `databases_demo_run`, `answers_demo_run.json`）。这有助于将不同运行的输出隔离开。
    - `parallel_requests=1`, `top_n_retrieval=3`, `llm_reranking_sample_size=3`: 为演示设置了较小的并发和处理数量，以加快运行速度。
    - `submission_file=False`: 不生成用于竞赛提交的特定格式文件。
- **初始化 `Pipeline`**:
    - `root_path`: 设置为 `setup_pipeline_demo_data` 返回的路径。
    - `subset_name`, `questions_file_name`, `pdf_reports_dir_name`: 指定在 `root_path` 下要使用的文件名和子目录名。
    - `run_config`: 传入我们创建的配置对象。

#### 4.3. 运行 Pipeline 各个阶段
脚本随后按顺序调用 `Pipeline` 实例的各个方法，模拟一个完整的 RAG 数据处理和问答流程：

1.  **`Pipeline.download_docling_models()`**: （静态方法）检查并下载 Docling 解析模型（如果本地没有）。
2.  **`pipeline.parse_pdf_reports_sequential()`**:
    -   解析 `subset_demo.csv` 中列出的 PDF（即 `democorp_pipeline_report.pdf`）。
    -   输出：在 `demo_data_root / debug_data / parsed_reports_json_demo_run /` 目录下生成 `democorp_pipeline_report.json` (原始解析结果)。
3.  **`pipeline.serialize_tables(max_workers=1)`**:
    -   对上一步生成的 `parsed_reports_json_demo_run` 目录中的 JSON 文件进行表格序列化（类似 `demo_04`）。这里设置为单 worker 运行。
    -   输出：直接修改 `parsed_reports_json_demo_run` 目录中的 JSON 文件，在表格对象中添加 `serialized` 字段。
4.  **`pipeline.process_parsed_reports()`**:
    -   这是一个非常核心的后处理步骤，它内部会执行多个子任务：
        -   **合并与简化 (Merging)**: 将每个页面的内容整合成单一文本字符串，并智能融入序列化表格信息（类似 `demo_05`）。输出到 `merged_reports_json_demo_run` 目录。
        -   **Markdown 导出 (MD Export)**: 将简化后的 JSON 报告导出为 Markdown 文件（类似 `demo_06`）。输出到 `reports_markdown_demo_run` 目录。
        -   **文本切块 (Chunking)**: 将简化报告中的文本切分成小块，并特殊处理序列化表格的信息块（类似 `demo_07`）。输出到 `chunked_reports_json_demo_run` 目录。
        -   **向量数据库创建 (Vector DB Creation)**:
            -   为所有文本块生成嵌入向量，并构建 FAISS 索引（类似 `demo_09`）。输出到 `databases_demo_run / faiss_indices /` 目录（例如 `democorp_pipeline_report.faiss`）。
            -   为所有文本块构建 BM25 索引（类似 `demo_10`）。输出到 `databases_demo_run / bm25_indices /` 目录（例如 `democorp_pipeline_report.bm25.pkl`）。
5.  **`pipeline.process_questions()`**:
    -   读取 `questions_demo.json` 中的问题。
    -   对每个问题，执行完整的 RAG 流程（类似 `demo_18` 或 `demo_20`）：
        -   确定目标公司/文档（"DemoCorp Pipeline"）。
        -   加载对应的 FAISS 和 BM25 索引。
        -   执行检索（可能是混合检索）。
        -   进行 LLM 重排序（因为 `llm_reranking=True`）。
        -   根据问题 `kind` 指定的 schema，调用 LLM 生成结构化答案。
    -   输出：将所有问题的答案汇总到一个 JSON 文件中，保存到 `demo_data_root` 目录下，文件名通常是 `answers_demo_run.json` (或带有数字后缀，如 `answers_demo_run_01.json`)。

#### 4.4. 显示结果与清理提示
-   **显示答案**: 脚本使用 `find_answers_file` 找到生成的答案 JSON 文件，加载并美化打印其内容。
-   **输出信息汇总**: 清晰地列出本次演示运行产生的各种中间数据和最终结果所在的目录路径。
-   **清理提示**: 提醒用户，如果需要，可以手动删除整个 `study/pipeline_demo_data` 目录来清理本次演示产生的所有文件。

## 关键启示

1.  **`Pipeline` 作为总指挥**: `Pipeline` 类将 RAG 的复杂流程分解为一系列可管理、可配置、可按序执行的阶段。
2.  **配置驱动 (`RunConfig`)**: 通过 `RunConfig` 可以灵活控制流程中的关键行为和参数，例如是否使用表格序列化、是否进行 LLM 重排序、输出路径的命名规则等。
3.  **端到端自动化**: 从原始 PDF 和问题列表输入，到最终生成结构化的答案输出，整个过程可以高度自动化。
4.  **数据流转清晰**: 脚本的输出和最后的总结清晰地展示了数据在各个处理阶段是如何被转换和存储的，以及中间产物和最终结果分别位于何处。
5.  **模块化与集成**: 这个 Demo 完美地展示了之前各个独立 Demo 中介绍的技术和组件是如何被无缝集成到一个统一的流程中的。

## 如何运行脚本

1.  **确保 `data/test_set/pdf_reports/194000c9109c6fa628f1fed33b44ae4c2b8365f4.pdf` 文件存在**，它是 `setup_pipeline_demo_data` 的源文件。
2.  **设置 `OPENAI_API_KEY` 环境变量**。
3.  **确保所有必要的库都已安装** (包括 `pandas`, `openai`, `python-dotenv`, `pydantic`, `faiss-cpu`, `rank_bm25` 等，以及 `src` 目录下所有自定义模块的依赖)。
4.  **打开终端或命令行工具**。
5.  **导航到脚本所在的目录**。
6.  **执行脚本**:
    ```bash
    python study/demo_21_simplified_pipeline_run.py
    ```
    或者，如果你在项目的根目录下：
    ```bash
    python -m study.demo_21_simplified_pipeline_run
    ```
    **注意**: 这个脚本会执行一个完整的 RAG 流程，包括多次 LLM API 调用（用于表格序列化、嵌入生成、重排序、问答等），因此**可能需要较长时间运行，并会产生相应的 API 调用费用**。
    运行完成后，仔细检查 `study/pipeline_demo_data/` 目录下的各个子目录和文件，特别是最终的答案 JSON 文件。

## 总结：RAG 系统的“一键启动”体验

`demo_21_simplified_pipeline_run.py` 是我们整个系列教程的集大成之作。它通过 `Pipeline` 类提供了一个“一键式”体验，让我们能够从一个高层次的视角观察一个完整的 RAG 系统是如何从原始文档和用户问题出发，经过一系列复杂的自动化处理步骤，最终生成智能答案的。

这不仅是对前面所有 Demo 内容的综合回顾与应用，也为我们提供了一个在实际项目中构建和管理复杂数据处理与 AI 应用流程的优秀范例。希望这个最终的演示能让你对 RAG 系统的全貌有一个更深刻、更完整的理解！感谢您坚持学习完整个系列教程！
