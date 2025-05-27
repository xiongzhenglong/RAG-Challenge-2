# RAG 实战核心：`demo_11_vector_retrieval.py` 之 FAISS 向量检索

大家好！在我们的 PDF 文档智能处理系列教程中，我们已经成功地为文档的文本块（chunks）创建了文本嵌入向量，并使用 FAISS 构建了一个高效的向量索引 (`demo_09`)。现在，万事俱备，只欠东风——如何利用这个 FAISS 索引来真正地“检索”与用户问题相关的信息呢？

本篇教程将通过 `study/demo_11_vector_retrieval.py` 脚本，向大家展示如何针对一个用户查询（sample query），执行向量检索（也常被称为“语义搜索”）。这是检索增强生成（RAG）系统中“R”——Retrieval（检索）环节的实战演练。

## 脚本目标

- 演示如何加载预先构建的 FAISS 索引（来自 `demo_09`）和对应的文本块数据（来自 `demo_07`）。
- 为用户提出的示例查询生成文本嵌入向量。
- 使用 FAISS 索引进行相似性搜索，找出与查询向量最相似的N个文本块。
- 展示检索到的文本块内容及其与查询的相似度（距离）信息。

## 什么是向量检索？

向量检索的核心思想是：**在向量空间中，根据语义相似度找到与用户查询最相关的内容。** 其典型流程如下：

1.  **为用户查询生成嵌入向量**:
    -   当用户输入一个自然语言查询时，系统首先使用**与构建文档索引时完全相同（或兼容）的嵌入模型**，将这个查询也转换成一个嵌入向量。这是确保“苹果”能与“苹果”进行比较的关键。
2.  **搜索向量索引**:
    -   将上一步生成的查询嵌入向量，在预先构建好的 FAISS 索引（或其他向量数据库）中进行搜索。
    -   FAISS 会高效地计算查询向量与索引中存储的所有文本块向量之间的“距离”或“相似度”。常用的距离度量有欧氏距离（L2 距离）或余弦相似度。
3.  **获取 Top-K 结果**:
    -   搜索操作会返回N个最相似的文本块的索引（indices）以及它们对应的距离/相似度分数。这些文本块被认为是与用户查询语义上最相关的内容。
4.  **构建上下文，生成答案**:
    -   这些检索到的 Top-K 文本块（通常连同原始查询）会一起被发送给大型语言模型（LLM）。LLM 利用这些上下文信息来生成一个更准确、更全面的答案。

## 前提条件

1.  **来自 `demo_07` 的切分后报告**: 位于 `study/chunked_reports_output/` 目录下的 JSON 文件（例如 `report_for_serialization.json`）。这个文件至关重要，因为它包含了 FAISS 索引中向量ID到实际文本块内容及其元数据的映射。
2.  **来自 `demo_09` 的 FAISS 索引文件**: 位于 `study/vector_dbs/` 目录下的 `.faiss` 文件（例如 `demo_report.faiss`）。这是我们进行搜索的目标索引。
3.  **`OPENAI_API_KEY` 环境变量**: **必须设置**，因为脚本需要调用 OpenAI API 来为用户查询生成嵌入向量。

## Python 脚本 `study/demo_11_vector_retrieval.py`

让我们完整地看一下这个脚本的代码：
```python
# study/demo_11_vector_retrieval.py

import json
import os
from pathlib import Path
import sys
import faiss
import numpy as np
from openai import OpenAI # For generating query embeddings
from dotenv import load_dotenv # For loading OPENAI_API_KEY from .env

# Load environment variables from .env file (especially OPENAI_API_KEY)
load_dotenv()

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """
    Demonstrates retrieving relevant text chunks from a FAISS vector database
    based on a sample query. This involves generating an embedding for the query
    and using FAISS to find the most similar chunk embeddings.
    """
    print("Starting vector retrieval demo...")

    # --- 1. Define Paths ---
    # Input chunked report (output of demo_07)
    input_chunked_report_dir = Path("study/chunked_reports_output/")
    input_chunked_filename = "report_for_serialization.json" # Assuming this name
    input_chunked_report_path = input_chunked_report_dir / input_chunked_filename

    # Input FAISS index (output of demo_09)
    faiss_index_dir = Path("study/vector_dbs/")
    faiss_index_filename = "demo_report.faiss" # Assuming this name from demo_09
    faiss_index_path = faiss_index_dir / faiss_index_filename

    print(f"Input chunked report path: {input_chunked_report_path}")
    print(f"FAISS index path: {faiss_index_path}")

    # --- 2. Prepare Data (Load Chunked JSON and FAISS Index) ---
    if not input_chunked_report_path.exists():
        print(f"Error: Input chunked JSON file not found at {input_chunked_report_path}")
        print("Please ensure 'demo_07_text_splitting.py' has run successfully.")
        return
    if not faiss_index_path.exists():
        print(f"Error: FAISS index file not found at {faiss_index_path}")
        print("Please ensure 'demo_09_creating_vector_db.py' has run successfully.")
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

    try:
        faiss_index = faiss.read_index(str(faiss_index_path))
        print(f"Successfully loaded FAISS index. Index contains {faiss_index.ntotal} vectors.")
        if faiss_index.ntotal != len(chunks):
            print(f"Warning: Number of vectors in FAISS index ({faiss_index.ntotal}) "
                  f"does not match number of chunks in JSON ({len(chunks)}). "
                  "This might lead to incorrect retrieval mapping.")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return

    # --- 3. Understanding Vector Retrieval ---
    # Vector retrieval is the process of finding the most relevant items (text chunks)
    # from a collection based on their semantic similarity to a user's query.
    # The process typically involves:
    #   1. Generating an Embedding for the Query: The user's query (natural language)
    #      is converted into a numerical vector (embedding) using the same embedding
    #      model that was used to create embeddings for the text chunks.
    #   2. Searching the Index: This query embedding is then used to search the
    #      vector database (FAISS index in this case). FAISS efficiently calculates
    #      the "distance" (e.g., L2 distance, cosine similarity) between the query
    #      embedding and all the chunk embeddings stored in the index.
    #   3. Retrieving Top-K Chunks: The search returns the indices of the top-K
    #      most similar chunks (those with the smallest distance or highest similarity).
    #      These chunks are considered the most semantically relevant to the query.
    # These retrieved chunks are then passed to an LLM as context to generate an answer.

    # --- 4. Perform Retrieval ---
    sample_query = "What were the total revenues?"
    print(f"\n--- Performing Vector Retrieval for Query: \"{sample_query}\" ---")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Query embedding generation requires an OpenAI API key.")
        return

    try:
        llm = OpenAI(api_key=openai_api_key, timeout=20.0, max_retries=2) # Standard client
        print("OpenAI client initialized.")

        # Generate query embedding
        print("Generating embedding for the query...")
        # Using a newer model like text-embedding-3-large or text-embedding-3-small is recommended.
        # Ensure the dimensionality matches the one used for indexing (ada-002 was 1536).
        # text-embedding-3-large default is 3072, text-embedding-3-small is 1536.
        # If demo_09 used ada-002 (1536 dims), use text-embedding-3-small or specify dimensions.
        # For this demo, let's assume text-embedding-3-small for matching ada-002's common dim if needed.
        # Or, if the index was created with text-embedding-3-large, this is fine.
        # It's crucial that query embedding model matches document embedding model/dimensions.
        # Let's use text-embedding-ada-002 for consistency with typical FAISS index dimension of 1536.
        # If your index was built with a different model/dimension, adjust here.
        embedding_model = "text-embedding-ada-002" # Or "text-embedding-3-small" for 1536 dims
        
        embedding_response = llm.embeddings.create(
            input=sample_query,
            model=embedding_model
        )
        query_embedding = embedding_response.data[0].embedding
        query_embedding_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        print(f"Query embedding generated. Shape: {query_embedding_np.shape}")

        # Search the FAISS index
        top_k = 5 # Number of top results to retrieve
        print(f"Searching FAISS index for top {top_k} similar chunks...")
        distances, indices = faiss_index.search(query_embedding_np, top_k)
        
        print("\n--- Retrieval Results ---")
        if not indices[0].size:
            print("No results found.")
        else:
            for i in range(len(indices[0])):
                retrieved_chunk_index = indices[0][i]
                retrieved_distance = distances[0][i]

                if retrieved_chunk_index < 0 or retrieved_chunk_index >= len(chunks):
                    print(f"  Result {i+1}: Invalid index {retrieved_chunk_index} returned by FAISS. Skipping.")
                    continue

                retrieved_chunk = chunks[retrieved_chunk_index]
                chunk_id = retrieved_chunk.get('id', 'N/A')
                page_num = retrieved_chunk.get('page_number', 'N/A')
                chunk_text_snippet = retrieved_chunk.get('text', '')[:250] # First 250 chars

                print(f"  Result {i+1}:")
                print(f"    Chunk ID: {chunk_id}")
                print(f"    Page Number: {page_num}")
                # FAISS L2 distance is non-negative; smaller is better.
                # For cosine similarity, FAISS often returns 1 - cosine_sim for IndexFlatIP, so smaller is better.
                # If using IndexFlatL2, distance is Euclidean distance.
                print(f"    Similarity Score (Distance): {retrieved_distance:.4f}")
                print(f"    Text Snippet: \"{chunk_text_snippet}...\"")
                print("-" * 20)

    except Exception as e:
        print(f"An error occurred during vector retrieval: {e}")
        import traceback
        traceback.print_exc()
    print("----------------------------------------------------")

    print("\nVector retrieval demo complete.")

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
import faiss                     # FAISS 库
import numpy as np               # NumPy 库
from openai import OpenAI        # OpenAI 官方 Python 库，用于调用嵌入模型 API
from dotenv import load_dotenv   # 用于从 .env 文件加载环境变量

load_dotenv() # 加载 .env 文件 (通常包含 OPENAI_API_KEY)
sys.path.append(...) # 添加 src 目录到 Python 路径
```
- `faiss` 和 `numpy` 用于加载和操作 FAISS 索引。
- `openai.OpenAI` 是 OpenAI 官方提供的与其 API 交互的客户端。
- `dotenv.load_dotenv()`: 这个函数会尝试从项目根目录下的 `.env` 文件中加载环境变量。这是一种推荐的做法，用于管理像 API 密钥这样的敏感信息，而不是将它们硬编码到脚本中。

### 2. 定义路径
```python
    input_chunked_report_path = Path("study/chunked_reports_output/") / "report_for_serialization.json"
    faiss_index_path = Path("study/vector_dbs/") / "demo_report.faiss"
```
- `input_chunked_report_path`: 指向 `demo_07` 输出的、包含所有文本块及其元数据的 JSON 文件。当 FAISS 返回相似向量的索引ID后，我们需要这个文件来查找对应索引ID的实际文本内容。
- `faiss_index_path`: 指向 `demo_09` 创建并保存的 FAISS 索引文件。

### 3. 准备数据（加载文本块 JSON 和 FAISS 索引）
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

    try:
        faiss_index = faiss.read_index(str(faiss_index_path)) # 加载 FAISS 索引
        print(f"Successfully loaded FAISS index. Index contains {faiss_index.ntotal} vectors.")
        if faiss_index.ntotal != len(chunks):
            print(f"Warning: Number of vectors in FAISS index ({faiss_index.ntotal}) "
                  f"does not match number of chunks in JSON ({len(chunks)}). "
                  "This might lead to incorrect retrieval mapping.")
    # ... (错误处理) ...
```
- **加载文本块**: 从 `input_chunked_report_path` 文件中加载 `content['chunks']` 列表到 `chunks` 变量。这个列表中的每个元素是一个字典，包含了文本块的 `id`, `text`, `page_number` 等信息。FAISS 返回的是向量在索引中的位置（0, 1, 2...），这个位置直接对应于 `chunks` 列表的索引。
- **加载 FAISS 索引**: 使用 `faiss.read_index(str(faiss_index_path))` 从磁盘加载之前保存的 FAISS 索引。
- **数量校验**: 一个重要的健全性检查是 `faiss_index.ntotal` (FAISS 索引中的向量总数) 是否等于 `len(chunks)` (JSON 文件中的文本块总数)。如果两者不匹配，意味着 FAISS 索引与其元数据（文本块内容）之间可能存在偏差，会导致检索结果映射错误。

### 4. 理解向量检索（脚本中的第 3 部分注释）
这部分注释详细解释了向量检索的步骤：为查询生成嵌入 -> 搜索索引 -> 获取 Top-K 结果。我们在教程开头已充分讨论。

### 5. 执行检索
这是脚本的核心交互部分：

#### 5.1. 定义示例查询和初始化 OpenAI 客户端
```python
    sample_query = "What were the total revenues?" # 示例用户查询
    print(f"\n--- Performing Vector Retrieval for Query: \"{sample_query}\" ---")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        # ... (API Key 检查与错误提示) ...
        return
    try:
        llm = OpenAI(api_key=openai_api_key, timeout=20.0, max_retries=2)
        print("OpenAI client initialized.")
    # ... (OpenAI 客户端初始化错误处理) ...
```
- `sample_query`: 一个示例性的用户自然语言查询。
- `openai_api_key = os.getenv("OPENAI_API_KEY")`: 从环境变量中获取 OpenAI API 密钥。
- `llm = OpenAI(...)`: 初始化 OpenAI 客户端，后续将用它来调用嵌入模型 API。

#### 5.2. 为查询生成嵌入向量
```python
        print("Generating embedding for the query...")
        embedding_model = "text-embedding-ada-002" # 或其他兼容模型
        
        embedding_response = llm.embeddings.create(
            input=sample_query,
            model=embedding_model
        )
        query_embedding = embedding_response.data[0].embedding
        query_embedding_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        print(f"Query embedding generated. Shape: {query_embedding_np.shape}")
```
- `embedding_model`: **极其重要的一点**！这里选择的嵌入模型（及其输出的维度）**必须**与 `demo_09` 中为文档块创建 FAISS 索引时使用的嵌入模型**完全一致或高度兼容**。
    - 如果 `demo_09` 使用的是 `text-embedding-ada-002`（输出1536维向量），那么这里也应该使用 `text-embedding-ada-002` 或其他输出同样1536维并语义上兼容的模型（如 `text-embedding-3-small`）。
    - 如果维度不匹配，FAISS 搜索将失败或产生无意义的结果。
- `llm.embeddings.create(...)`: 调用 OpenAI API 的 `embeddings.create` 端点，传入查询文本和指定的模型，获取嵌入向量。
- `query_embedding_np = np.array(...).reshape(1, -1)`:
    - 将返回的嵌入向量（一个 Python 列表）转换为 NumPy 数组。
    - `.astype(np.float32)` 确保数据类型为单精度浮点。
    - `.reshape(1, -1)` 将其转换为一个行向量（即一个 1xN 的二维数组），因为 FAISS 的 `search` 方法期望查询向量是这种格式。

#### 5.3. 搜索 FAISS 索引
```python
        top_k = 5 # 希望检索到的最相关结果数量
        print(f"Searching FAISS index for top {top_k} similar chunks...")
        distances, indices = faiss_index.search(query_embedding_np, top_k)
```
- `top_k`: 定义了我们希望从 FAISS 索引中检索回的最相似的文本块的数量。
- `distances, indices = faiss_index.search(query_embedding_np, top_k)`: **核心的 FAISS 搜索调用**。
    - `query_embedding_np`: 上一步生成的查询嵌入向量（1xN NumPy 数组）。
    - `top_k`: 要返回的结果数量。
    - **返回值**:
        - `distances`: 一个二维 NumPy 数组，形状为 `(1, top_k)`。包含了查询向量与每个返回结果向量之间的计算出的“距离”。距离越小，通常表示越相似（具体取决于 FAISS 索引的构建方式，如 `IndexFlatL2` 使用L2距离）。
        - `indices`: 一个二维 NumPy 数组，形状也为 `(1, top_k)`。包含了在 FAISS 索引中与查询向量最相似的 `top_k` 个向量的**索引ID**（从0开始，对应于它们被添加到索引时的顺序）。这些索引ID直接对应于我们之前加载的 `chunks` 列表中的元素位置。

#### 5.4. 显示检索结果
```python
        print("\n--- Retrieval Results ---")
        if not indices[0].size: # 检查是否有结果返回
            print("No results found.")
        else:
            for i in range(len(indices[0])):
                retrieved_chunk_index = indices[0][i] # 获取向量在 FAISS 中的索引 ID
                retrieved_distance = distances[0][i]  # 获取对应的距离

                if retrieved_chunk_index < 0 or retrieved_chunk_index >= len(chunks):
                    # ... (无效索引处理，理论上不应发生除非索引与chunks列表不匹配) ...
                    continue

                retrieved_chunk = chunks[retrieved_chunk_index] # 使用索引ID从chunks列表中获取原始块数据
                chunk_id = retrieved_chunk.get('id', 'N/A')
                page_num = retrieved_chunk.get('page_number', 'N/A')
                chunk_text_snippet = retrieved_chunk.get('text', '')[:250]

                print(f"  Result {i+1}:")
                print(f"    Chunk ID: {chunk_id}")
                print(f"    Page Number: {page_num}")
                print(f"    Similarity Score (Distance): {retrieved_distance:.4f}") # 打印距离
                print(f"    Text Snippet: \"{chunk_text_snippet}...\"")
                print("-" * 20)
```
- 脚本遍历 FAISS 返回的 `indices[0]` 列表（因为我们只查询了一个向量，所以结果在第一个元素中）。
- `retrieved_chunk_index = indices[0][i]`: 这是 FAISS 返回的向量在其内部索引中的位置。**这个位置直接对应于 `chunks` 列表中的索引**。
- `retrieved_chunk = chunks[retrieved_chunk_index]`: 通过这个索引，我们可以从 `chunks` 列表中取回该向量对应的原始文本块的所有信息（ID, 文本内容, 页码等）。
- **显示信息**: 打印出每个检索到的块的ID、来源页码、与查询的距离（相似度得分）以及文本片段。

### 6. 解读距离/相似度得分
- 如果 FAISS 索引是使用 `IndexFlatL2`（如 `demo_09` 中可能的方式）构建的，那么 `distances` 返回的是**欧氏距离（L2 距离）**。这种情况下，距离值越小，表示文本块与查询的语义越相似。
- 如果使用的是 `IndexFlatIP`（内积，通常对应余弦相似度），FAISS 为了保持“距离越小越好”的惯例，有时会返回 `1 - cosine_similarity`。或者，如果直接返回内积，则值越大越相似。你需要了解你的 FAISS 索引是如何构建和归一化的来正确解读这个分数。

## 如何运行脚本

1.  **确保 `demo_07` 和 `demo_09` 已成功运行**:
    - `study/chunked_reports_output/report_for_serialization.json` (来自 `demo_07`) 必须存在。
    - `study/vector_dbs/demo_report.faiss` (来自 `demo_09`) 必须存在。
2.  **设置 `OPENAI_API_KEY` 环境变量**: **至关重要！**
    ```bash
    export OPENAI_API_KEY="sk-YourActualOpenAIKeyHere" 
    ```
    (或者在项目根目录创建一个 `.env` 文件并写入 `OPENAI_API_KEY=sk-...`)
3.  **确保相关库已安装**: `pip install faiss-cpu openai python-dotenv numpy` (或 `faiss-gpu` 如果适用)。
4.  **打开终端或命令行工具**。
5.  **导航到脚本所在的目录**。
6.  **执行脚本**:
    ```bash
    python study/demo_11_vector_retrieval.py
    ```
    或者，如果你在项目的根目录下：
    ```bash
    python -m study.demo_11_vector_retrieval
    ```
    脚本将为示例查询生成嵌入，并使用 FAISS 索引检索出最相关的文本块，然后打印它们的信息。

## 总结：实现 RAG 的“检索”核心

`demo_11_vector_retrieval.py` 为我们清晰地展示了向量检索（语义搜索）的实际操作过程。通过为用户查询生成嵌入，并在 FAISS 索引中查找最相似的文本块嵌入，我们能够从大量文档内容中精确地定位到与用户意图最相关的片段。

这些检索到的文本块是 RAG 系统生成高质量、有依据答案的关键上下文。理解并能实现这一检索步骤，意味着你已经掌握了构建 RAG 应用中最核心的技术之一。希望这篇教程能帮助你成功实践向量检索！
