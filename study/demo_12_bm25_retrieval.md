# RAG 实战核心：`demo_12_bm25_retrieval.py` 之 BM25 关键词检索

大家好！在我们的 PDF 文档智能处理系列教程中，我们已经探索了两种主要的文本检索技术：
- `demo_11` 展示了如何利用 FAISS 索引进行**语义检索（密集向量检索）**，它能理解查询和文本块之间的深层含义相似性。
- 而在更早的 `demo_10` 中，我们构建了另一种重要的索引——**BM25 索引**，它为**关键词检索（稀疏检索）**奠定了基础。

本篇教程将通过 `study/demo_12_bm25_retrieval.py` 脚本，向大家演示如何实际使用 `demo_10` 中创建的 BM25 索引来响应用户查询，找出最相关的文本块。这将清晰地展示稀疏检索的工作流程。

## 脚本目标

- 演示如何加载预先构建和序列化（pickle）的 BM25 索引对象（来自 `demo_10`）以及对应的文本块数据（来自 `demo_07`）。
- 对用户提出的示例查询进行分词处理。
- 使用加载的 BM25 模型计算语料库中每个文本块与查询的相关性得分。
- 检索并展示得分最高的N个文本块及其 BM25 分数。

## 什么是 BM25 检索？

BM25 检索是一种基于关键词匹配的成熟算法。其核心流程如下：

1.  **查询分词 (Query Tokenization)**:
    -   用户的自然语言查询首先会被分解成一系列单独的词语（tokens）。这个过程通常包括转小写、去除标点符号、按空格分割等简单操作。
2.  **文档（文本块）评分**:
    -   加载的 BM25 模型（例如 `rank_bm25` 库中的 `BM25Okapi` 对象）会针对分词后的查询，计算索引语料库中每一个文档（在我们的例子中是“文本块”）的相关性得分。
    -   这个得分主要基于以下因素：
        -   **词频 (Term Frequency, TF)**: 查询词在某个文本块中出现的频率。
        -   **逆文档频率 (Inverse Document Frequency, IDF)**: 查询词在整个文本块集合中的稀有程度。越稀有的词（即在少数几个文本块中出现的词）通常被赋予更高的权重。
        -   **文档长度归一化**: BM25 算法会考虑到文本块的长度，对较长的文本块（可能仅仅因为长而碰巧匹配到更多词）进行一定的“惩罚”或调整，使得评分更公平。
3.  **获取 Top-K 结果**:
    -   所有文本块根据其 BM25 得分进行降序排列。
    -   得分最高的N个文本块被认为是与查询关键词最相关的结果。

BM25 对于那些用户明确知道要搜索哪些关键词的场景非常有效。

## 前提条件

1.  **来自 `demo_07` 的切分后报告**: 位于 `study/chunked_reports_output/` 目录下的 JSON 文件（例如 `report_for_serialization.json`）。这个文件包含了所有文本块的原文和元数据，是检索结果的最终内容来源。
2.  **来自 `demo_10` 的 BM25 索引文件**: 位于 `study/bm25_indices/` 目录下的 `.bm25.pkl` 文件（例如 `report_for_serialization.bm25.pkl`）。这是序列化保存的、已经“训练”好的 BM25 模型对象。

## Python 脚本 `study/demo_12_bm25_retrieval.py`

让我们完整地看一下这个脚本的代码：
```python
# study/demo_12_bm25_retrieval.py

import json
import os
from pathlib import Path
import sys
import pickle # For loading the BM25 index object

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Note: BM25Ingestor is not directly needed here if we are just loading a pickled BM25Okapi object.
# However, the BM25Okapi object itself would have been created using a library like `rank_bm25`.

def main():
    """
    Demonstrates retrieving relevant text chunks using a pre-built BM25 index.
    This involves tokenizing a query, using the BM25 model to score chunks,
    and retrieving the top-scoring ones.
    """
    print("Starting BM25 retrieval demo...")

    # --- 1. Define Paths ---
    # Input chunked report (output of demo_07)
    input_chunked_report_dir = Path("study/chunked_reports_output/")
    input_chunked_filename = "report_for_serialization.json" # Assuming this name
    input_chunked_report_path = input_chunked_report_dir / input_chunked_filename

    # Input BM25 index (output of demo_10)
    bm25_index_dir = Path("study/bm25_indices/")
    bm25_index_filename = input_chunked_filename.replace(".json", ".bm25.pkl") # From demo_10
    bm25_index_path = bm25_index_dir / bm25_index_filename

    print(f"Input chunked report path: {input_chunked_report_path}")
    print(f"BM25 index path: {bm25_index_path}")

    # --- 2. Prepare Data (Load Chunked JSON and BM25 Index) ---
    if not input_chunked_report_path.exists():
        print(f"Error: Input chunked JSON file not found at {input_chunked_report_path}")
        print("Please ensure 'demo_07_text_splitting.py' has run successfully.")
        return
    if not bm25_index_path.exists():
        print(f"Error: BM25 index file not found at {bm25_index_path}")
        print("Please ensure 'demo_10_creating_bm25_index.py' has run successfully.")
        return

    chunks = []
    try:
        with open(input_chunked_report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        if 'content' in report_data and 'chunks' in report_data['content'] and report_data['content']['chunks']:
            chunks = report_data['content']['chunks']
            if not chunks:
                print("No chunks found in the loaded JSON file. Cannot perform retrieval.")
                return
            print(f"Successfully loaded chunked JSON. Found {len(chunks)} chunks.")
        else:
            print("Error: 'content' or 'chunks' not found in the loaded JSON structure.")
            return
    except json.JSONDecodeError:
        print(f"Error: Could not decode the JSON file at {input_chunked_report_path}.")
        return
    except Exception as e:
        print(f"An error occurred while loading the chunked JSON: {e}")
        return

    bm25_index = None
    try:
        with open(bm25_index_path, 'rb') as f:
            bm25_index = pickle.load(f) # This should be a BM25Okapi instance (or similar)
        print(f"Successfully loaded BM25 index from {bm25_index_path}")
        # Basic check if the loaded object has a 'get_scores' method
        if not hasattr(bm25_index, 'get_scores'):
            print("Error: Loaded BM25 index object does not have a 'get_scores' method.")
            print("Ensure it's a valid BM25 model (e.g., from rank_bm25 library).")
            return
    except Exception as e:
        print(f"Error loading BM25 index: {e}")
        return

    # --- 3. Understanding BM25 Retrieval ---
    # BM25 retrieval works as follows:
    #   1. Tokenize the Query: The input query string is broken down into individual
    #      words (tokens), often after lowercasing and basic cleaning.
    #   2. Score Documents (Chunks): The BM25 algorithm calculates a relevance score
    #      for each document (chunk) in the indexed corpus with respect to the
    #      tokenized query. This score is based on:
    #         - Term Frequency (TF): How often query terms appear in a document.
    #         - Inverse Document Frequency (IDF): How rare or common the query terms
    #           are across the entire corpus of documents. Rare terms get higher weight.
    #         - Document Length: BM25 penalizes very long documents that might match
    #           terms by chance, and normalizes for document length.
    #   3. Retrieve Top-N Chunks: The documents are ranked by their BM25 scores,
    #      and the top N highest-scoring documents are returned as the most relevant results.
    # This method is effective for keyword-based searches.

    # --- 4. Perform Retrieval ---
    sample_query = "What were the company's main risks?"
    print(f"\n--- Performing BM25 Retrieval for Query: \"{sample_query}\" ---")

    try:
        # Tokenize the query (simple lowercasing and splitting)
        tokenized_query = sample_query.lower().split()
        print(f"Tokenized query: {tokenized_query}")

        # Get scores from the BM25 index
        # The `get_scores` method of a BM25Okapi object (from rank_bm25 library)
        # takes the tokenized query and returns an array of scores, one for each document
        # in the corpus that the BM25 model was trained on.
        print("Calculating BM25 scores for all chunks...")
        doc_scores = bm25_index.get_scores(tokenized_query)
        
        if len(doc_scores) != len(chunks):
            print(f"Warning: Number of scores ({len(doc_scores)}) from BM25 "
                  f"does not match number of chunks ({len(chunks)}).")
            print("This indicates a mismatch between the indexed corpus and the loaded chunks.")
            # We can still proceed but results might be misaligned if the original corpus changed.

        # Get top N results
        top_k = 5
        # Sort indices by score in descending order
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        print(f"\n--- Top {top_k} BM25 Retrieval Results ---")
        if not top_indices:
            print("No results found or scores were all zero/negative.")
        else:
            for i, retrieved_chunk_index in enumerate(top_indices):
                # Ensure the index is valid for the loaded chunks list
                if retrieved_chunk_index < 0 or retrieved_chunk_index >= len(chunks):
                    print(f"  Result {i+1}: Invalid index {retrieved_chunk_index} from BM25 sort. Skipping.")
                    continue

                retrieved_chunk = chunks[retrieved_chunk_index]
                bm25_score = doc_scores[retrieved_chunk_index]
                
                chunk_id = retrieved_chunk.get('id', 'N/A')
                page_num = retrieved_chunk.get('page_number', 'N/A')
                chunk_text_snippet = retrieved_chunk.get('text', '')[:250] # First 250 chars

                print(f"  Result {i+1}:")
                print(f"    Chunk ID: {chunk_id}")
                print(f"    Page Number: {page_num}")
                print(f"    BM25 Score: {bm25_score:.4f}") # Higher is generally better
                print(f"    Text Snippet: \"{chunk_text_snippet}...\"")
                print("-" * 20)

    except Exception as e:
        print(f"An error occurred during BM25 retrieval: {e}")
        import traceback
        traceback.print_exc()
    print("----------------------------------------------------")

    print("\nBM25 retrieval demo complete.")

if __name__ == "__main__":
    main()
```

## 脚本代码详解

### 1. 导入模块
```python
import json
import os
from pathlib import Path
import sys
import pickle # 用于加载（反序列化）BM25 索引对象
sys.path.append(...)
# 注意: BM25Ingestor 在这里不是必需的，因为我们直接加载 pickle 文件
```
- `pickle`: 用于从磁盘加载之前在 `demo_10` 中序列化保存的 BM25 模型对象。

### 2. 定义路径
```python
    input_chunked_report_path = Path("study/chunked_reports_output/") / "report_for_serialization.json"
    bm25_index_path = Path("study/bm25_indices/") / "report_for_serialization.bm25.pkl"
```
- `input_chunked_report_path`: 指向 `demo_07` 输出的、包含所有文本块及其元数据的 JSON 文件。当 BM25 返回相关文本块的索引ID后，我们需要这个文件来查找对应ID的实际文本内容。
- `bm25_index_path`: 指向 `demo_10` 创建并保存的 BM25 索引文件（`.bm25.pkl`）。

### 3. 准备数据（加载文本块 JSON 和 BM25 索引）
```python
    # ... (检查文件是否存在) ...
    chunks = []
    try:
        with open(input_chunked_report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        if 'content' in report_data and 'chunks' in report_data['content'] and report_data['content']['chunks']:
            chunks = report_data['content']['chunks'] # 加载所有文本块数据
            # ... (检查 chunks 是否为空) ...
    # ... (错误处理) ...

    bm25_index = None
    try:
        with open(bm25_index_path, 'rb') as f: # 以二进制读模式 'rb' 打开
            bm25_index = pickle.load(f) # 反序列化 BM25 对象
        print(f"Successfully loaded BM25 index from {bm25_index_path}")
        if not hasattr(bm25_index, 'get_scores'): # 检查是否是有效的 BM25 对象
            # ... (错误提示) ...
            return
    # ... (错误处理) ...
```
- **加载文本块**: 从 `input_chunked_report_path` 文件中加载 `content['chunks']` 列表到 `chunks` 变量。这个列表中的每个元素是一个字典，包含了文本块的 `id`, `text`, `page_number` 等信息。BM25 返回的是文本块在原始语料库中的索引位置，这个位置直接对应于 `chunks` 列表的索引。
- **加载 BM25 索引**:
    - 使用 `with open(bm25_index_path, 'rb') as f:` 以**二进制读模式 (`'rb'`)** 打开 `.bm25.pkl` 文件。
    - `bm25_index = pickle.load(f)`: 使用 `pickle.load()` 从文件中反序列化出之前保存的 BM25 模型对象（例如，一个 `rank_bm25.BM25Okapi` 类的实例）。
- **有效性检查**: `if not hasattr(bm25_index, 'get_scores')`: 这是一个简单的检查，判断加载的对象是否拥有 `get_scores` 方法。这是 `rank_bm25` 库中 BM25 模型对象的典型方法，用于计算文档得分。

### 4. 理解 BM25 检索（脚本中的第 3 部分注释）
这部分注释详细解释了 BM25 检索的步骤：查询分词 -> 文档评分 (基于 TF, IDF, 文档长度) -> 获取 Top-K 结果。我们在教程开头已充分讨论。

### 5. 执行检索
这是脚本的核心交互部分：

#### 5.1. 定义示例查询并进行分词
```python
    sample_query = "What were the company's main risks?" # 示例用户查询
    print(f"\n--- Performing BM25 Retrieval for Query: \"{sample_query}\" ---")

    try:
        tokenized_query = sample_query.lower().split() # 简单的分词处理
        print(f"Tokenized query: {tokenized_query}")
```
- `sample_query`: 一个示例性的用户自然语言查询。
- `tokenized_query = sample_query.lower().split()`: **查询预处理**。
    - `.lower()`: 将查询转换为小写，这通常是为了与索引构建时的文本处理保持一致（大部分 BM25 实现会对语料库文本也做小写处理）。
    - `.split()`: 按空格将查询字符串分割成一个词语列表（tokens）。这是一个非常基础的分词方法。在实际应用中，可能会使用更复杂的分词器，特别是对于中文等语言。**重要的是，查询的分词方式应与构建 BM25 索引时对文档进行分词的方式保持一致或兼容。**

#### 5.2. 计算所有文本块的 BM25 得分
```python
        print("Calculating BM25 scores for all chunks...")
        doc_scores = bm25_index.get_scores(tokenized_query) # 获取所有文档的得分
        
        if len(doc_scores) != len(chunks):
            # ... (数量不匹配的警告) ...
```
- `doc_scores = bm25_index.get_scores(tokenized_query)`: **核心的 BM25 评分调用**。
    - `bm25_index` (即加载的 `BM25Okapi` 对象) 的 `get_scores` 方法接收分词后的查询 `tokenized_query`。
    - 它会为构建索引时使用的语料库中的**每一个文档（文本块）**计算一个相关性得分。
    - 返回的 `doc_scores` 是一个 NumPy 数组（或类似列表的结构），其长度应等于原始文本块的数量，每个元素是对应文本块的 BM25 分数。

#### 5.3. 获取 Top-K 结果
```python
        top_k = 5 # 希望检索到的最相关结果数量
        # 通过得分对索引进行排序，获取得分最高的 top_k 个索引
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
```
- `top_k`: 定义了我们希望从结果中选取的得分最高的文本块数量。
- `top_indices = sorted(...)`:
    - `range(len(doc_scores))` 生成一个从 0 到 `len(doc_scores)-1` 的索引序列。
    - `key=lambda i: doc_scores[i]` 指定排序时依据 `doc_scores` 数组中对应索引 `i` 的值（即 BM25 分数）进行排序。
    - `reverse=True` 表示按降序排列（BM25 分数越高越相关）。
    - `[:top_k]` 取排序后的前 `top_k` 个索引。
    - `top_indices` 现在是一个列表，包含了得分最高的 `top_k` 个文本块在原始 `chunks` 列表（也是 `doc_scores` 列表）中的索引位置。

#### 5.4. 显示检索结果
```python
        print(f"\n--- Top {top_k} BM25 Retrieval Results ---")
        if not top_indices:
            print("No results found or scores were all zero/negative.")
        else:
            for i, retrieved_chunk_index in enumerate(top_indices):
                # ... (检查 retrieved_chunk_index 是否有效) ...

                retrieved_chunk = chunks[retrieved_chunk_index] # 使用索引从 chunks 列表中获取原始块数据
                bm25_score = doc_scores[retrieved_chunk_index]  # 获取对应的 BM25 分数
                
                # ... (获取 chunk_id, page_num, chunk_text_snippet) ...

                print(f"  Result {i+1}:")
                print(f"    Chunk ID: {chunk_id}")
                print(f"    Page Number: {page_num}")
                print(f"    BM25 Score: {bm25_score:.4f}") # 打印 BM25 分数
                print(f"    Text Snippet: \"{chunk_text_snippet}...\"")
                print("-" * 20)
```
- 脚本遍历 `top_indices` 列表。
- `retrieved_chunk_index`: 这是文本块在原始 `chunks` 列表中的索引。
- `retrieved_chunk = chunks[retrieved_chunk_index]`: 通过这个索引，我们可以从 `chunks` 列表中取回该文本块的所有信息。
- `bm25_score = doc_scores[retrieved_chunk_index]`: 获取该文本块的 BM25 得分。
- **显示信息**: 打印出每个检索到的块的ID、来源页码、BM25 得分以及文本片段。

### 6. 解读 BM25 得分
对于 BM25 算法，通常来说，**得分越高，表示文本块与查询的关键词匹配度越高，即越相关**。这些得分是浮点数，其绝对值大小取决于语料库和查询本身。

## 如何运行脚本

1.  **确保 `demo_07` 和 `demo_10` 已成功运行**:
    - `study/chunked_reports_output/report_for_serialization.json` (来自 `demo_07`) 必须存在。
    - `study/bm25_indices/report_for_serialization.bm25.pkl` (来自 `demo_10`) 必须存在。
2.  **确保相关库已安装**: `pip install rank_bm25` (如果 `BM25Ingestor` 或加载的 pickle 对象依赖它) 和 `pip install numpy`。
3.  **打开终端或命令行工具**。
4.  **导航到脚本所在的目录**。
5.  **执行脚本**:
    ```bash
    python study/demo_12_bm25_retrieval.py
    ```
    或者，如果你在项目的根目录下：
    ```bash
    python -m study.demo_12_bm25_retrieval
    ```
    脚本将为示例查询进行分词，并使用加载的 BM25 索引检索出最相关的文本块，然后打印它们的信息及 BM25 得分。

## 总结：关键词检索的力量

`demo_12_bm25_retrieval.py` 向我们展示了如何利用预先构建的 BM25 索引执行高效的关键词检索。通过对用户查询进行分词，并利用 BM25 模型计算每个文本块的相关性得分，我们可以快速找到那些在关键词层面与查询最匹配的内容。

BM25 是一种久经考验的、效果显著的稀疏检索方法。在 RAG 系统中，它可以作为语义检索（如 `demo_11` 中基于 FAISS 的检索）的有力补充，甚至可以通过混合检索策略（结合两者的结果）来进一步提升整体检索性能。

到此，我们关于 PDF 文档解析、处理、不同类型索引构建及检索的系列教程就全部结束了。希望这些详细的演示能帮助你掌握构建自己的文档智能应用所需的关键技术和流程！
