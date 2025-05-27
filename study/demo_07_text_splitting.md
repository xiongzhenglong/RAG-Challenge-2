# RAG 系统的基石：`demo_07_text_splitting.py` 之文本切分详解

大家好！经过前面一系列教程的学习，我们已经将原始 PDF 文档处理成了包含丰富信息、每页内容被整合为单一文本字符串的简化版 JSON 报告（`demo_05` 的产物）。现在，我们距离将这些信息喂给大型语言模型（LLM）又近了一步。但在此之前，还有一个至关重要的步骤——**文本切分（Text Splitting 或 Chunking）**。

本篇教程将通过 `study/demo_07_text_splitting.py` 脚本，向大家展示如何使用 `src.text_splitter.TextSplitter` 类对我们处理好的报告文本进行切分。这对于构建检索增强生成（RAG）系统尤其关键。

## 脚本目标

- 演示如何使用 `TextSplitter` 将来自合并后 JSON 报告的文本内容切分成更小的、易于管理的数据块（chunks）。
- 解释文本切分为何对 RAG 系统至关重要。
- 特别展示 `TextSplitter` 如何将 `demo_04` 中 LLM 生成的“序列化表格信息块”作为独立的、高质量的文本块进行处理。
- 介绍切分后数据的结构。

## 什么是文本切分（Chunking）？

想象一下，你想让 LLM 阅读一本很厚的书并回答相关问题。如果一次性把整本书都丢给模型，它可能会“消化不良”。LLM 的“工作记忆”（即上下文窗口 context window）是有限的。因此，我们需要把长文本切分成更小的片段或“块”（chunks）。

**文本切分的重要性主要体现在：**

1.  **突破上下文窗口限制**: LLM 一次能处理的文本长度有限。长文档必须切分才能被完整处理。
2.  **精准检索**: 在 RAG 系统中，当用户提问时，系统需要从大量文档中找到最相关的片段来辅助生成答案。小而精的文本块能显著提高检索的准确性。
3.  **提升效率**: 处理小文本块比处理整个文档更快，消耗资源也更少。

`TextSplitter` 类（很可能借鉴了 LangChain 中 `RecursiveCharacterTextSplitter` 的思想）采用一种策略，尝试根据文本的自然边界（如段落、换行符、空格）进行切分，以尽可能保持语义的完整性。关键参数包括：

-   `chunk_size`: 每个文本块的理想最大长度（例如，按字符数或 token 数计算）。
-   `chunk_overlap`: 相邻文本块之间的一小部分重叠内容。这有助于在块与块之间保持上下文的连续性，防止信息在边界处丢失。

**本 Demo 的亮点：智能处理序列化表格**
此 `TextSplitter` 不仅仅是对 `demo_05` 输出的页面纯文本进行盲目切分。它有一个非常重要的特性：能够识别并利用 `demo_04` 中 `TableSerializer` 生成的、经过 LLM精心构造的表格“信息块”（`information_blocks`）。它会将这些信息块作为**独立的、高质量的文本块**提取出来，而不是简单地将表格的 Markdown 或 HTML 表示混在普通文本中一起切分。这样做的好处是，每个关于表格的信息块本身就是一段上下文丰富、语义完整的自然语言描述，非常适合直接用于 RAG。

## 前提条件

此脚本依赖于前两个 demo 的输出：

1.  **来自 `demo_05` 的合并报告**: 位于 `study/merged_reports_output/` 目录下的 JSON 文件（例如 `report_for_serialization.json`）。这是主要的文本来源，其中每页内容已被整合为单一字符串。
2.  **来自 `demo_04` 的含序列化表格的报告**: 位于 `study/temp_serialization_data/` 目录下的 JSON 文件（同名，例如 `report_for_serialization.json`）。`TextSplitter` 会从这个文件中读取 `tables[X]['serialized']['information_blocks']`，以便将它们作为单独的块处理。

## Python 脚本 `study/demo_07_text_splitting.py`

让我们完整地看一下这个脚本的代码：
```python
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
```

## 脚本代码详解

### 1. 导入模块
```python
import json
import os
import shutil
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.text_splitter import TextSplitter # 核心：文本切分器
```
- `TextSplitter` 是执行文本切分任务的核心类。

### 2. 定义路径
```python
    # 输入1: demo_05 输出的合并报告
    input_merged_reports_dir = Path("study/merged_reports_output/")
    input_merged_filename = "report_for_serialization.json"
    input_merged_full_path = input_merged_reports_dir / input_merged_filename

    # 输入2: demo_04 输出的含序列化表格的报告
    serialized_tables_input_dir = Path("study/temp_serialization_data/")
    serialized_tables_filename = "report_for_serialization.json"
    serialized_tables_full_path = serialized_tables_input_dir / serialized_tables_filename

    # 输出: 切分后的报告
    chunked_output_dir = Path("study/chunked_reports_output/")
    chunked_output_path = chunked_output_dir / input_merged_filename
```
- `input_merged_full_path`: 指向 `demo_05` 生成的合并简化后的 JSON 文件。这是主要的文本切分对象。
- `serialized_tables_full_path`: 指向 `demo_04` 生成的包含序列化表格（即 `information_blocks`）的 JSON 文件。`TextSplitter` 会从此文件中提取这些 `information_blocks` 作为高质量的独立文本块。
- `chunked_output_dir` 和 `chunked_output_path`: 定义了存放切分后 JSON 报告的目录和文件名。文件名通常与输入文件名保持一致。

### 3. 准备输入数据
```python
    if not input_merged_full_path.exists():
        # ... 错误处理 ...
        return
    if not serialized_tables_full_path.exists():
        # ... 错误处理 ...
        return
    chunked_output_dir.mkdir(parents=True, exist_ok=True)
```
- 脚本会检查上述两个关键的输入文件是否存在，确保有数据可供处理。
- 创建用于存放切分后报告的输出目录 `chunked_output_dir`。

### 4. 理解文本切分（脚本中的第 3 部分注释）
这部分注释详细解释了文本切分的目的、方法及其对 RAG 系统的重要性。关键点包括：
- **为何切分**: LLM 上下文窗口限制、精准检索需求、处理效率。
- **切分策略**: `TextSplitter` 很可能使用类似 LangChain 的 `RecursiveCharacterTextSplitter`，尝试按语义边界（如段落、换行）切分。
- **`chunk_size` 和 `chunk_overlap`**: 控制块大小和块间重叠。
- **亮点：独立处理序列化表格**: `TextSplitter` 会将 `demo_04` 中生成的表格 `information_blocks` 作为独立的、高质量的文本块来处理，而不是将它们混在页面文本中一起被常规方法切分。这能最大程度保留表格摘要的上下文和语义完整性。

### 5. 执行切分
```python
    print("\nInitializing TextSplitter and processing reports...")
    splitter = TextSplitter(
        # chunk_size=1000, # 默认值通常是合理的
        # chunk_overlap=200, # 默认值通常是合理的
    )

    try:
        splitter.split_all_reports(
            all_report_dir=input_merged_reports_dir,
            output_dir=chunked_output_dir,
            serialized_tables_dir=serialized_tables_input_dir
        )
        print("Text splitting process complete.")
    except Exception as e:
        # ... 错误处理 ...
        return
```
- **初始化 `TextSplitter`**:
    - `splitter = TextSplitter(...)`: 创建 `TextSplitter` 对象。脚本中注释掉了 `chunk_size` 和 `chunk_overlap` 参数，这意味着它可能会使用类内部定义的默认值（例如，常见的 1000 字符块大小和 100-200 字符的重叠）。在实际应用中，这些参数可能需要根据具体任务和 LLM 的特性进行调整。
- **执行切分**:
    - `splitter.split_all_reports(...)`: 这是执行切分的核心方法。
        - `all_report_dir`: 指向包含合并后报告（来自 `demo_05`）的目录。`TextSplitter` 会处理这个目录下的所有 JSON 文件。
        - `output_dir`: 切分后的结果（新的 JSON 文件，其中包含文本块列表）将保存到这个目录。
        - `serialized_tables_dir`: **关键参数**！指向包含序列化表格报告（来自 `demo_04`）的目录。`TextSplitter` 在处理 `all_report_dir` 中的某个文件时，会去 `serialized_tables_dir` 查找同名的文件，并从中提取 `tables[X]['serialized']['information_blocks']`。这些 `information_blocks` 会被视为独立的、高质量的文本块。
    - 该方法内部逻辑大致是：
        1.  遍历 `all_report_dir` 中的每个报告文件。
        2.  对于每个报告，加载其内容（主要是 `page['text']`）。
        3.  同时，加载 `serialized_tables_dir` 中对应的报告文件，提取所有表格的 `information_blocks`。
        4.  将每个 `information_block` 作为一个独立的文本块（chunk）。
        5.  对每个页面的 `page['text']` 内容（在移除了已作为独立块处理的表格部分之后，或者如果表格未被序列化处理，则包含表格的 Markdown/HTML）应用文本切分算法（如 `RecursiveCharacterTextSplitter`），生成更多文本块。
        6.  将所有生成的文本块（来自页面文本和独立表格信息块）汇总，并为每个块添加元数据（如块 ID、来源页码、块类型等）。
        7.  将包含这些文本块列表的新 JSON 数据结构保存到 `output_dir`。

### 6. 加载并显示切分后的报告数据
```python
    print("\n--- Chunked Report Data ---")
    # ... (检查 chunked_output_path 是否存在) ...
    try:
        with open(chunked_output_path, 'r', encoding='utf-8') as f:
            chunked_data = json.load(f)

        # 5.1. Metainfo (应被保留)
        # ... (打印元信息) ...

        # 5.2. Content Structure
        if 'content' in chunked_data:
            if 'pages' in chunked_data['content']: # 原页面结构可能保留
                # ...
            if 'chunks' in chunked_data['content']: # 核心：文本块列表
                num_chunks = len(chunked_data['content']['chunks'])
                print(f"    `content['chunks']` found: Total {num_chunks} chunks generated.")

                # 5.3. Details of First Few Chunks
                for i, chunk in enumerate(chunked_data['content']['chunks'][:3]):
                    print(f"    --- Chunk {i+1} ---")
                    print(f"      ID: {chunk.get('id')}")
                    print(f"      Type: {chunk.get('type')} (e.g., 'content' or 'serialized_table')")
                    print(f"      Page: {chunk.get('page_number')}")
                    print(f"      Length (tokens): {chunk.get('length_tokens')}")
                    text_snippet = chunk.get('text', '')[:200]
                    print(f"      Text Snippet: \"{text_snippet}...\"")
    # ... (错误处理) ...
```
- **加载新文件**: 脚本加载由 `TextSplitter` 生成的位于 `chunked_output_path` 的新 JSON 文件。
- **元信息**: `metainfo` 部分通常会被保留。
- **内容结构变化 (`content`)**:
    - `content['pages']`: 原有的页面结构（来自 `demo_05`）可能仍然被保留在输出中，这取决于 `TextSplitter` 的具体实现，有时保留原始页面信息有助于追溯块的来源。
    - `content['chunks']`: **这是最重要的部分**。它是一个列表，包含了所有从文档中切分出来的文本块。
- **单个文本块 (Chunk) 的结构**: 列表中每个 `chunk` 通常是一个字典，包含以下信息：
    - `id`: 块的唯一标识符。
    - `type`: 块的类型。可能是 `'content'`（来自页面的常规文本切分）或 `'serialized_table'`（直接来自 `demo_04` 中表格的 `information_blocks`）。
    - `page_number`: 该块内容来源于原始文档的哪个页码。
    - `length_tokens`: 块的长度，通常用 token 数量估算（LLM 处理文本的基本单位）。
    - `text`: **实际的文本块内容**。如果 `type` 是 `'serialized_table'`，这里的 `text` 就是一个完整的 `information_block`。
- 脚本会打印出前几个文本块的这些详细信息，让我们能直观地看到切分的结果，特别是那些类型为 `serialized_table` 的高质量表格信息块。

### 7. 清理（可选）
脚本末尾提供了删除 `chunked_output_dir` 临时目录的可选代码。

## 为何这种切分对 RAG 特别有效？

1.  **高质量的表格信息块**: 将 LLM 生成的表格 `information_blocks` 作为独立的块，保证了这些高度浓缩、上下文丰富的表格摘要能够被完整地索引和检索。在回答与表格内容相关的问题时，直接检索到这些摘要远比检索到原始表格的 Markdown 或 HTML 然后再让 LLM 理解要高效和准确得多。
2.  **语义相关的文本块**: `RecursiveCharacterTextSplitter` 等方法尽可能在语义边界（如段落）处切分，保持了文本的连贯性。
3.  **大小适中**: 切分后的块大小适合 LLM 的上下文窗口，便于模型处理。
4.  **明确的来源**: 每个块都带有来源页码等元数据，方便追溯和引用。

## 如何运行脚本

1.  **确保 `demo_04` 和 `demo_05` 已成功运行**:
    - `study/temp_serialization_data/report_for_serialization.json` (来自 `demo_04`) 必须存在。
    - `study/merged_reports_output/report_for_serialization.json` (来自 `demo_05`) 必须存在。
2.  **打开终端或命令行工具**。
3.  **导航到脚本所在的目录**。
4.  **执行脚本**:
    ```bash
    python study/demo_07_text_splitting.py
    ```
    或者，如果你在项目的根目录下：
    ```bash
    python -m study.demo_07_text_splitting
    ```
    脚本执行后，检查 `study/chunked_reports_output/report_for_serialization.json` 文件，仔细观察 `content['chunks']` 列表及其中的每个块的结构，特别是那些 `type: 'serialized_table'` 的块。

## 总结

`demo_07_text_splitting.py` 和 `TextSplitter` 为我们展示了在将文档内容送入 LLM（尤其是 RAG 系统）之前的最后一个关键预处理步骤——文本切分。通过智能地将长文本（包括对序列化表格的特殊处理）分解成大小合适、语义连贯、上下文丰富的文本块，我们可以显著提升后续检索和生成任务的效率与质量。

这些精心准备的文本块是构建高效 RAG 系统的基石，它们可以直接被用于生成向量嵌入（embeddings）并存入向量数据库中，等待用户的查询。希望这篇教程能帮助你理解文本切分的重要性及其实现方式！
