# RAG 终极挑战：`demo_20_processing_comparative_question.py` 之比较型问题全流程处理

大家好！欢迎来到我们 PDF 文档智能处理与检索增强生成（RAG）系列教程的真正压轴大戏——`demo_20_processing_comparative_question.py`！在此前的教程中，我们已经：
- 掌握了 RAG 的各项核心组件技术，从文档解析、索引构建（FAISS `demo_09`, BM25 `demo_10`）、不同类型的检索（`demo_11`, `demo_12`）、LLM 重排序 (`demo_16`) 到与 LLM API 的直接交互（`demo_14` 获取文本，`demo_15` 获取结构化JSON）。
- 在 `demo_18` 中，我们见证了 `QuestionsProcessor` 如何为单个、直接的问题（非比较型）自动化整个 RAG 流程。
- 在 `demo_19` 中，我们学习了如何将比较型问题预处理，分解成针对每个比较对象的独立子问题。

现在，`demo_20` 将把这些能力——特别是问题分解和端到端处理——结合起来，展示 `QuestionsProcessor` 如何智能地处理一个**比较型问题**，从最初的复杂查询，到检索各方信息，再到最终生成一个综合性的比较答案。

## 脚本目标

- 演示 `QuestionsProcessor` 如何端到端地处理一个涉及多个实体（例如公司）的比较型问题。
- 理解处理比较型问题的完整 RAG 流程：
    1.  为演示准备特定于每个比较实体的数据（独立的 JSON 报告和 FAISS 索引）。
    2.  `QuestionsProcessor` 内部自动进行实体识别和问题改写。
    3.  为每个改写的子问题独立执行 RAG（检索、重排序、信息提取）。
    4.  最后，综合所有收集到的信息，生成一个针对原始比较型问题的、结构化的比较答案。
- 展示最终输出的比较型答案的结构。

## RAG 系统如何应对比较型问题的挑战？

比较型问题（例如“A公司和B公司在2023年的收入分别是多少？哪家更高？”）对 RAG 系统提出了更高的要求，因为它们：
-   涉及多个信息焦点（A公司的收入，B公司的收入）。
-   需要分别获取这些焦点的信息。
-   最后还需要对这些信息进行比较和综合判断。

简单地将整个比较型问题直接扔给检索引擎，可能无法高效、准确地同时找到所有相关信息。

**`QuestionsProcessor` 的策略（概念性）：**

`QuestionsProcessor` 在处理指定了 `schema="comparative"` 的比较型问题时，通常会采用一种“分而治之再汇总”的策略：

1.  **实体识别与问题改写 (Entity Identification & Question Rephrasing)**:
    -   首先，从原始比较型问题中识别出需要进行比较的关键实体（例如，"AlphaCorp", "BetaInc"）。这可能通过 LLM 或其他 NLP 技术完成。
    -   然后，利用 LLM（类似于 `demo_19` 的机制）将原始比较型问题分解成针对每个实体的、更具体、更聚焦的子问题。例如：
        -   "AlphaCorp 在2023年的总收入是多少？"
        -   "BetaInc 在2023年的总收入是多少？"
2.  **独立信息获取 (Individual Answering/Fact Extraction)**:
    -   对于每一个改写后的子问题，`QuestionsProcessor` 会独立地为该实体执行一次完整的 RAG 流程：
        -   **检索(Retrieve)**: 从该实体的专属数据源（例如，AlphaCorp 的报告对应的 FAISS 索引）中检索最相关的文本块。
        -   **重排序(Rerank)**: （如果启用）对检索到的文本块进行 LLM 重排序。
        -   **信息提取(Extract)**: 将筛选后的上下文和子问题一起提交给 LLM，要求其根据一个简单的 schema（如 "number" 或 "text"）提取或生成针对该子问题的具体答案/信息。
3.  **最终答案综合 (Final Synthesis)**:
    -   收集所有针对子问题的答案/信息（例如，AlphaCorp的收入是$X，BetaInc的收入是$Y）。
    -   最后，将这些收集到的独立信息片段，连同**原始的比较型问题**，一起提交给 LLM。
    -   此时，会使用一个专门为比较型问题设计的 Prompt（例如，`demo_13` 中讨论的 `ComparativeAnswerPrompt`，由 `schema="comparative"` 触发）和对应的 Pydantic 输出模型。
    -   LLM 被要求基于所有提供的独立事实，对原始比较型问题给出一个综合性的、比较性的答案（例如，“BetaInc的收入（$Y）高于AlphaCorp的收入（$X）...”）。

## 为比较型 Demo 特别准备数据：`prepare_comparative_demo_data` 函数

为了让这个比较型问题的 Demo 能够清晰地工作，我们需要为每个参与比较的公司（AlphaCorp, BetaInc）准备独立的、包含其特定信息的模拟数据源。这就是 `prepare_comparative_demo_data` 函数的作用。

-   **输入**: 一个基础的、通用的已切块 JSON 报告模板路径 (`base_chunked_json_template_path`)，新的切块报告存放目录，新的向量数据库目录，公司名，以及要为该公司注入的特定文本信息（例如，包含其收入数据的文本）。
-   **核心操作**:
    1.  **创建公司专属 JSON 报告**:
        -   复制基础模板 JSON 文件，并以公司名（小写）命名（例如 `alphacorp.json`, `betainc.json`），存放到新的 `chunked_reports_dir`。
        -   修改这个新的 JSON 文件：
            -   设置 `metainfo['company_name']` 为当前公司名。
            -   设置 `metainfo['sha1_name']` 为公司名（小写），这将用于关联对应的 FAISS 索引文件。
            -   **注入特定信息**: 将包含该公司特定数据（如收入）的 `revenue_text` 添加到该 JSON 文件第一个文本块的开头。这确保了当我们查询该公司信息时，能从其专属文档中找到特定数据。
    2.  **为公司专属报告创建 FAISS 索引**:
        -   使用 `VectorDBIngestor`（需要 `OPENAI_API_KEY` 来生成嵌入向量）。
        -   读取刚刚为该公司创建并修改的 JSON 文件中的所有文本块。
        -   为这些文本块生成嵌入向量。
        -   构建一个新的 FAISS 索引。
        -   将这个 FAISS 索引保存到新的 `vector_dbs_dir` 目录下，并以公司名（小写，即 `sha1_name`）命名（例如 `alphacorp.faiss`, `betainc.faiss`）。

通过这个函数，我们为 AlphaCorp 和 BetaInc 分别创建了包含它们各自收入信息的独立数据集和检索引擎。这是成功演示比较型问答的关键前提。

## 前提条件

1.  **基础模板 JSON 文件**: 一个来自 `demo_07` 的输出文件（例如 `study/chunked_reports_output/report_for_serialization.json`），作为创建公司专属数据的模板。
2.  **`OPENAI_API_KEY` 环境变量**: 因为数据准备（生成嵌入）和后续的 RAG 流程（问题改写、信息提取、答案综合）都需要调用 OpenAI API。

## Python 脚本 `study/demo_20_processing_comparative_question.py`

让我们完整地看一下这个脚本的代码：
```python
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
```

## 脚本代码详解

### 1. 导入模块
```python
# ... (标准导入) ...
import shutil # 用于文件复制
import faiss  # 用于创建 FAISS 索引
import numpy as np # 用于处理 FAISS 所需的 NumPy 数组

from src.questions_processing import QuestionsProcessor # 核心：问题处理器
from src.ingestion import VectorDBIngestor # 用于为每个公司数据创建 FAISS 索引
```
- 新增了 `shutil`, `faiss`, `numpy` 和 `VectorDBIngestor` 的导入，因为 `prepare_comparative_demo_data` 函数会为每个公司动态创建其专属的 JSON 数据和 FAISS 索引。

### 2. `prepare_comparative_demo_data` 函数
这个函数是本 Demo 的关键准备步骤，确保每个被比较的公司都有自己独立的、包含特定信息的报告和检索引擎。
- **参数**:
    - `base_chunked_json_template_path`: 一个通用的、已切块的 JSON 文件路径，作为模板。
    - `chunked_reports_dir`, `vector_dbs_dir`: 分别是存放为本 demo 新生成的公司特定 JSON 文件和 FAISS 索引的目录。
    - `company_name`: 当前要为其准备数据的公司名称。
    - `revenue_text`: 包含该公司特定收入信息的文本字符串。
    - `overwrite`: 是否覆盖已存在的文件。
- **核心逻辑**:
    1.  **创建公司专属 JSON**: 复制模板 JSON，以公司名（小写）命名（如 `alphacorp.json`）。
    2.  **修改元数据**: 在新 JSON 的 `metainfo` 中设置 `company_name` 和 `sha1_name`（`sha1_name` 通常是公司名小写，用于关联 FAISS 文件）。
    3.  **注入特定信息**: 将 `revenue_text` 添加到此公司 JSON 文件的第一个文本块中，确保该公司的数据包含其特有的收入信息。
    4.  **创建专属 FAISS 索引**:
        -   使用 `VectorDBIngestor`（需要 OpenAI API Key 生成嵌入）。
        -   读取刚修改过的公司专属 JSON 中的所有文本块。
        -   为这些文本块生成嵌入向量。
        -   构建一个新的 FAISS 索引。
        -   将此 FAISS 索引保存到 `vector_dbs_dir`，并以公司名（小写，即 `sha1_name`）命名（如 `alphacorp.faiss`）。
- **结果**: 对每个公司（AlphaCorp, BetaInc）调用此函数后，我们将在 `study/comparative_demo_data/` 下得到它们各自的 JSON 文件（包含各自的收入信息）和 FAISS 索引。

### 3. `main()` 函数

#### 3.1. 定义路径和公司特定数据
```python
    base_template_json_path = Path("study/chunked_reports_output/report_for_serialization.json")
    chunked_reports_dir = Path("study/comparative_demo_data/chunked_reports/") # 新的专用目录
    vector_dbs_dir = Path("study/comparative_demo_data/vector_dbs/")     # 新的专用目录
    
    company1_name = "AlphaCorp"
    company1_revenue_text = "AlphaCorp's total revenue in 2023 was $500 million."
    company2_name = "BetaInc"
    company2_revenue_text = "BetaInc's total revenue in 2023 was $750 million."
```
- 注意这里为比较型问题演示使用了新的数据目录 `study/comparative_demo_data/`，以避免与之前的 demo 数据混淆。
- 为 AlphaCorp 和 BetaInc 分别定义了包含其2023年收入的文本。

#### 3.2. 检查 API 密钥并准备演示数据
```python
    # ... (API Key 检查) ...
    if not base_template_json_path.exists(): # 确保模板存在
        # ...
        return
    print("Preparing data for AlphaCorp...")
    if not prepare_comparative_demo_data(... company1_name, company1_revenue_text ...):
        return
    print("\nPreparing data for BetaInc...")
    if not prepare_comparative_demo_data(... company2_name, company2_revenue_text ...):
        return
```
- 在执行任何操作前，会为 `AlphaCorp` 和 `BetaInc` 分别调用 `prepare_comparative_demo_data`，确保它们各自拥有包含特定收入信息的 JSON 文件和对应的 FAISS 索引。

#### 3.3. 初始化 `QuestionsProcessor`
```python
    processor = QuestionsProcessor(
        vector_db_dir=vector_dbs_dir,         # 指向包含 AlphaCorp.faiss, BetaInc.faiss 的目录
        documents_dir=chunked_reports_dir,    # 指向包含 AlphaCorp.json, BetaInc.json 的目录
        llm_reranking=True,
        parent_document_retrieval=False, # 保持简单，不使用父子块检索
        api_provider="openai",
        new_challenge_pipeline=False # 避免特定路径依赖
    )
```
- `QuestionsProcessor` 现在使用为比较型演示专门准备的 `vector_db_dir` 和 `documents_dir`。
- `llm_reranking=True` 意味着在为每个公司的子问题检索上下文后，会进行一轮 LLM 重排序。

#### 3.4. 处理比较型问题
```python
    comparative_question = (
        f"Which company had higher total revenue in 2023: '{company1_name}' or '{company2_name}'? "
        "Provide the revenue for each and state which one was higher."
    )
    schema = "comparative" # <--- 指定使用比较型问题的处理流程和输出模式

    # ... (打印问题和期望 schema) ...
    try:
        answer_dict = processor.process_question(
            question=comparative_question, 
            schema=schema
            # companies_for_comparison=[company1_name, company2_name] # 可选参数
        )
```
- `comparative_question`: 一个明确的比较型问题。
- `schema = "comparative"`: **这是关键**。当 `schema` 设置为 `"comparative"` 时，`QuestionsProcessor` 会启用其内部为处理比较型问题设计的特殊逻辑：
    1.  **实体识别**: 从 `comparative_question` 中识别出 "AlphaCorp" 和 "BetaInc"。 (如果 `companies_for_comparison` 参数被显式提供，则直接使用。)
    2.  **问题改写**: 调用 LLM（类似 `demo_19` 的机制）将原始问题分解为针对 AlphaCorp 和 BetaInc 的独立子问题，例如：
        - "What was the total revenue for AlphaCorp in 2023?"
        - "What was the total revenue for BetaInc in 2023?"
    3.  **独立信息获取**: 对每个子问题，`QuestionsProcessor` 会：
        -   找到对应公司的 JSON 报告（例如 `alphacorp.json`）和 FAISS 索引（`alphacorp.faiss`）。
        -   执行检索（向量搜索 + 可能的BM25）获取相关文本块。
        -   进行 LLM 重排序（因为 `llm_reranking=True`）。
        -   将最佳上下文和子问题传给 LLM，要求其根据一个简单的 schema（例如 "number" 或 "text"）提取具体信息（例如 AlphaCorp 的收入）。
    4.  **答案综合**: 将从上一步获得的关于 AlphaCorp 和 BetaInc 的独立信息（例如，AlphaCorp 收入 $500M，BetaInc 收入 $750M）收集起来。然后，将这些信息连同**原始的 `comparative_question`** 一起，再次提交给 LLM。这次会使用一个专门为生成比较型答案设计的 Prompt（例如 `src.prompts.ComparativeAnswerPrompt`）和对应的 Pydantic 输出模型。LLM 会基于提供的事实进行比较并生成最终的综合性答案。
- `answer_dict`: `process_question` 返回的 Python 字典，其结构应符合 `ComparativeAnswerPrompt` 关联的 Pydantic 模型。

#### 3.5. 显示处理结果
```python
        if answer_dict:
            print("  Full Answer Dictionary (from Pydantic model, e.g., ComparativeAnswer):")
            print(json.dumps(answer_dict, indent=2, ensure_ascii=False))
            
            print("\n  Key Extracted Information:")
            print(f"    Final Comparative Answer: {answer_dict.get('final_answer', 'N/A')}")
            # ... (打印 step_by_step_analysis 和 individual_answers) ...
            # ... (显示最终LLM综合调用的元数据) ...
```
- `answer_dict` 的预期结构（由 `ComparativeAnswerPrompt` 的 Pydantic 模型定义）可能包含：
    - `final_answer`: 对原始比较型问题的直接回答，例如 "BetaInc had higher total revenue in 2023 ($750 million) compared to AlphaCorp ($500 million)."
    - `step_by_step_analysis`: LLM 进行比较和得出结论的思考过程。
    - `individual_answers`: 一个列表，其中每个元素是一个字典，包含了针对每个公司（AlphaCorp, BetaInc）的独立子问题的答案和相关信息（如提取到的收入、使用的上下文片段等）。这有助于追溯比较结果的来源。
- 脚本会打印整个字典，并单独提取一些关键信息进行展示。

#### 3.6. 清理提示
提醒用户本 demo 在 `study/comparative_demo_data/` 目录下创建了新的公司特定文件。

## 关键启示

1.  **自动化处理复杂比较**: `QuestionsProcessor` 通过编排多个步骤（问题改写、独立检索与信息提取、最终综合）来自动化处理复杂的比较型问题。
2.  **数据准备的重要性**: 对于需要比较不同实体的情况，确保每个实体都有其独立的、包含准确信息的数据源和检索引擎是至关重要的。`prepare_comparative_demo_data` 函数模拟了这个过程。
3.  **Schema 驱动的端到端流程**: 通过 `schema="comparative"` 参数，整个流程被引导使用特定的 Prompts 和 Pydantic 模型，最终产出结构化的比较型答案。
4.  **模块化与可扩展性**: `QuestionsProcessor` 内部集成了检索、重排序、LLM 调用等模块，展现了 RAG 系统的模块化设计思想。

## 如何运行脚本

1.  **确保 `demo_07` 的输出 `study/chunked_reports_output/report_for_serialization.json` 存在**，它将作为创建公司特定数据的模板。
2.  **设置 `OPENAI_API_KEY` 环境变量**。
3.  **确保所有必要的库都已安装**。
4.  **打开终端或命令行工具**。
5.  **导航到脚本所在的目录**。
6.  **执行脚本**:
    ```bash
    python study/demo_20_processing_comparative_question.py
    ```
    或者，如果你在项目的根目录下：
    ```bash
    python -m study.demo_20_processing_comparative_question
    ```
    脚本将会：
    - 为 AlphaCorp 和 BetaInc 分别创建包含其特定收入信息的 JSON 文件和 FAISS 索引。
    - 初始化 `QuestionsProcessor`。
    - 对示例的比较型问题执行完整的 RAG 流程。
    - 打印出 LLM 返回的结构化比较答案字典以及相关的处理信息。
    由于涉及多次 LLM 调用（数据准备时的嵌入生成、问题改写、独立信息提取、最终答案综合），此脚本可能需要较长时间运行。

## 总结：RAG 能力的综合展现

`demo_20_processing_comparative_question.py` 是我们整个 RAG 系列教程的集大成者。它不仅整合了前面几乎所有 Demo 中介绍的技术和组件，还特别展示了 RAG 系统处理复杂、多方面比较型问题的强大能力。通过自动化的“分而治之”和“综合汇总”策略，结合 LLM 的深度理解和生成能力，我们可以构建出能够就复杂议题提供深入、有据比较的智能问答系统。

这真正开启了超越简单事实检索的、更高级的文档智能应用的大门。感谢您一路跟随本系列教程的学习，希望这些演示能为您的 AI 之旅提供坚实的基础和丰富的灵感！
