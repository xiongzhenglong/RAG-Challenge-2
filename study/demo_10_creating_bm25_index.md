# RAG 数据准备番外篇：`demo_10_creating_bm25_index.py` 之构建 BM25 稀疏检索索引

大家好！在我们的 PDF 文档智能处理系列教程中，我们已经深入探讨了如何将文档内容最终转化为适合语义搜索的 FAISS 向量索引 (`demo_09`)。但语义搜索（基于密集向量嵌入）并非 RAG 系统中唯一的检索利器。传统的基于关键词的检索方法，在某些场景下依然表现出色，甚至可以与语义搜索互为补充，形成更强大的混合检索系统。

本篇教程将通过 `study/demo_10_creating_bm25_index.py` 脚本，介绍另一种重要的检索索引技术——**BM25（Okapi BM25）**。我们将学习如何为 `demo_07` 中生成的文本块构建 BM25 索引，这是一种**稀疏检索（Sparse Retrieval）**方法。

## 脚本目标

- 演示如何使用 `src.ingestion.BM25Ingestor` 类为文本块创建 BM25 索引。
- 解释 BM25 的基本原理及其在信息检索中的作用。
- 对比 BM25（稀疏检索）与基于嵌入的密集向量检索（如 FAISS）。
- 将创建的 BM25 索引对象序列化并保存到磁盘。

## 什么是 BM25 (Okapi BM25)？

BM25 (Best Matching 25) 是一种成熟的**排序函数 (ranking function)**，广泛应用于搜索引擎中，用于评估文档集合中各个文档与用户查询的相关性。它是一种**词袋模型 (bag-of-words)** 的检索方法，依据查询中的词语在每个文档中出现的频率等统计信息对文档进行排序。

**BM25 的关键特性：**

1.  **基于词频 (Term Frequency-based)**:
    -   与依赖语义嵌入的密集向量检索不同，BM25 主要关注查询词在文档（在我们的案例中是“文本块”）中出现的频率 (TF)。
    -   它还考虑这些词在整个文档集合中的逆文档频率 (Inverse Document Frequency, IDF)，即一个词在越少的文档中出现，其重要性越高。
2.  **稀疏检索 (Sparse Retrieval)**:
    -   BM25 通常作用于文本的稀疏向量表示（例如，TF-IDF 向量）。“稀疏”意味着向量中的大多数元素为零，只有包含特定词语的维度才有非零值。
    -   它侧重于**精确的关键词匹配**及其统计显著性。
3.  **关键词匹配**: BM25 非常擅长找出那些包含了与用户查询完全相同的关键词的文档。

**与密集向量检索（例如，使用 FAISS 和文本嵌入）的对比：**

| 特性         | BM25 (稀疏检索)                                     | 密集向量检索 (如 FAISS + Embeddings)                     |
| :----------- | :---------------------------------------------------- | :------------------------------------------------------- |
| **核心机制** | 基于关键词频率 (TF-IDF) 和统计                         | 基于文本的语义理解和向量空间中的距离                     |
| **相似性**   | 关键词重叠度                                          | 语义相似度（即使词语不同但意思相近也能匹配）             |
| **向量表示** | 稀疏向量                                              | 密集向量 (所有维度都有值)                                |
| **擅长场景** | 精确关键词匹配，用户明确知道要搜什么词时               | 概念匹配，同义词/近义词搜索，理解用户意图                |
| **可能缺陷** | 可能错过语义相关但用词不同的文档；对表达方式敏感       | 对于非常 spécifiques 或罕见的关键词可能不如 BM25 精确      |

**常见应用场景：**

-   **基准系统 (Baseline)**: BM25 常常作为信息检索任务的一个强有力的基准对比模型。
-   **混合检索 (Hybrid Retrieval)**: 这是目前非常流行的做法——将 BM25 与密集检索方法结合起来。目标是利用两者的优势：BM25 的关键词匹配精度和密集检索的语义理解能力。可以将两者的得分进行融合（例如，使用倒数排序融合 Reciprocal Rank Fusion, RRF），以产生最终的、更优的文档排序结果。

`BM25Ingestor` 类很可能使用了像 `rank_bm25` 这样的 Python 库来在后台创建和管理 BM25 索引。

## 前提条件

1.  **来自 `demo_07` 的切分后报告**: 你需要先成功运行 `study/demo_07_text_splitting.py`。该脚本的输出（位于 `study/chunked_reports_output/` 目录下的 JSON 文件）包含了所有文本块，是本脚本的直接输入。BM25 将基于这些文本块的内容构建索引。

## Python 脚本 `study/demo_10_creating_bm25_index.py`

让我们完整地看一下这个脚本的代码：
```python
# study/demo_10_creating_bm25_index.py

import json
import os
from pathlib import Path
import sys
import pickle # For saving the BM25 index object

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import BM25Ingestor

def main():
    """
    Demonstrates creating a BM25 sparse retrieval index from the text chunks
    of a processed report.
    """
    print("Starting BM25 index creation demo...")

    # --- 1. Define Paths ---
    # Input is the chunked report (output of demo_07)
    input_chunked_report_dir = Path("study/chunked_reports_output/")
    input_chunked_filename = "report_for_serialization.json" # Assuming this name
    input_chunked_report_path = input_chunked_report_dir / input_chunked_filename

    # Output directory for the BM25 index
    bm25_output_dir = Path("study/bm25_indices/")
    # Determine BM25 index filename
    bm25_index_filename = input_chunked_filename.replace(".json", ".bm25.pkl")
    bm25_index_path = bm25_output_dir / bm25_index_filename

    print(f"Input chunked report path: {input_chunked_report_path}")
    print(f"BM25 index output directory: {bm25_output_dir}")
    print(f"BM25 index will be saved to: {bm25_index_path}")

    # --- 2. Prepare Input Data (Load Chunked JSON) ---
    if not input_chunked_report_path.exists():
        print(f"Error: Input chunked JSON file not found at {input_chunked_report_path}")
        print("Please ensure 'demo_07_text_splitting.py' has run successfully.")
        return

    all_chunk_texts = []
    try:
        with open(input_chunked_report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        if 'content' in report_data and 'chunks' in report_data['content'] and report_data['content']['chunks']:
            # Extract just the text content of each chunk
            all_chunk_texts = [chunk['text'] for chunk in report_data['content']['chunks'] if chunk.get('text')]
            if not all_chunk_texts:
                 print("No text content found in any chunks. Cannot create BM25 index.")
                 return
            print(f"Successfully loaded chunked JSON. Found {len(all_chunk_texts)} text chunks.")
        else:
            print("Error: 'content' or 'chunks' not found in the loaded JSON structure.")
            print("Please ensure the input file is correctly formatted (output of demo_07).")
            return
            
    except json.JSONDecodeError:
        print(f"Error: Could not decode the JSON file at {input_chunked_report_path}.")
        return
    except Exception as e:
        print(f"An error occurred while loading the JSON file: {e}")
        return

    # --- 3. Understanding BM25 (Okapi BM25) ---
    # BM25 (Best Matching 25) is a ranking function used by search engines to estimate
    # the relevance of documents to a given search query. It's a bag-of-words retrieval
    # function that ranks a set of documents based on the query terms appearing in each document.
    #
    # Key Characteristics:
    #   - Term Frequency-based: Unlike dense vector retrieval (which uses semantic embeddings),
    #     BM25 relies on the frequency of query terms within the documents (chunks, in our case)
    #     and the inverse document frequency (IDF) of those terms across the entire corpus.
    #   - Sparse Retrieval: It operates on sparse vector representations of text (e.g., TF-IDF vectors),
    #     focusing on exact keyword matches and their statistical importance.
    #   - Keyword Matching: It excels at finding documents that contain the exact keywords from the query.
    #
    # Contrast with Dense Vector Retrieval (e.g., using FAISS with embeddings):
    #   - Dense Retrieval: Captures semantic similarity. Can find relevant documents even if they
    #     don't use the exact query keywords but discuss similar concepts. Uses dense vector embeddings.
    #   - BM25 (Sparse Retrieval): Relies on keyword overlap. May miss documents that are semantically
    #     related but use different terminology.
    #
    # Common Use Cases:
    #   - Baseline: Often used as a strong baseline for information retrieval tasks.
    #   - Hybrid Retrieval: Frequently combined with dense retrieval methods. The idea is to leverage
    #     the strengths of both: BM25's keyword matching precision and dense retrieval's semantic understanding.
    #     Scores from both systems can be combined (e.g., using reciprocal rank fusion) to produce a final ranking.
    #
    # The `BM25Ingestor` likely uses a library like `rank_bm25` to create the index.

    # --- 4. Create BM25 Index ---
    print("\nInitializing BM25Ingestor and creating the BM25 index...")

    try:
        ingestor = BM25Ingestor() # This might initialize the BM25 model (e.g., BM25Okapi from rank_bm25)
        print("BM25Ingestor initialized.")

        print(f"Creating BM25 index from {len(all_chunk_texts)} text chunks...")
        # The `create_bm25_index` method would take the list of text chunks
        # and fit the BM25 model to this corpus.
        bm25_index = ingestor.create_bm25_index(all_chunk_texts)

        if not bm25_index: # Basic check
            print("Error: BM25 index creation failed or returned an invalid object.")
            return
            
        print(f"BM25 index created successfully.")
        # Note: The BM25 index object itself (e.g., a BM25Okapi instance) doesn't
        # typically have a direct '.ntotal' like FAISS. Its "size" is implicit
        # in the corpus it was trained on.

        # Create the output directory if it doesn't exist
        bm25_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {bm25_output_dir}")

        # Save the BM25 index to disk using pickle
        # BM25 models (like those from rank_bm25) are typically Python objects
        # and can be serialized using pickle.
        print(f"Saving BM25 index to: {bm25_index_path}...")
        with open(bm25_index_path, 'wb') as f:
            pickle.dump(bm25_index, f)
        print(f"BM25 index successfully saved to {bm25_index_path}")

    except Exception as e:
        print(f"An error occurred during BM25 index creation or saving: {e}")
        import traceback
        traceback.print_exc()
    print("----------------------------------------------------")

    print("\nBM25 index creation demo complete.")
    print("The generated .pkl file contains the BM25 model/index, ready for keyword-based searches.")

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
import pickle                     # 用于序列化和反序列化 Python 对象
from src.ingestion import BM25Ingestor # 核心：BM25 索引构建器
```
- `pickle`: Python 标准库，用于将 Python 对象（如训练好的 BM25 模型实例）序列化（保存）到磁盘，以及之后反序列化（加载）回来使用。
- `BM25Ingestor`: 负责创建 BM25 索引的类。

### 2. 定义路径
```python
    input_chunked_report_path = Path("study/chunked_reports_output/") / "report_for_serialization.json"
    bm25_output_dir = Path("study/bm25_indices/")
    bm25_index_filename = input_chunked_filename.replace(".json", ".bm25.pkl")
    bm25_index_path = bm25_output_dir / bm25_index_filename
```
- `input_chunked_report_path`: 指向 `demo_07` 输出的、包含所有文本块的 JSON 文件。
- `bm25_output_dir`: 指定存放生成的 BM25 索引文件的目录。
- `bm25_index_filename` 和 `bm25_index_path`: 定义了 BM25 索引文件的名称和完整路径。使用 `.bm25.pkl` 作为扩展名，表明它是一个通过 pickle 序列化的 BM25 索引对象。

### 3. 准备输入数据（加载所有文本块的文本内容）
```python
    # ... (检查 input_chunked_report_path 是否存在) ...
    all_chunk_texts = []
    try:
        with open(input_chunked_report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        if 'content' in report_data and 'chunks' in report_data['content'] and report_data['content']['chunks']:
            all_chunk_texts = [chunk['text'] for chunk in report_data['content']['chunks'] if chunk.get('text')]
            # ... (检查 all_chunk_texts 是否为空) ...
        # ... (错误处理) ...
    # ... (错误处理) ...
```
- 与 `demo_09` 类似，脚本加载 `demo_07` 的输出。
- `all_chunk_texts = [chunk['text'] for ... if chunk.get('text')]`: **关键区别**在于，这里只提取了每个文本块的实际文本内容 (`chunk['text']`) 形成一个字符串列表。BM25 算法直接作用于这些原始文本。

### 4. 理解 BM25（脚本中的第 3 部分注释）
这部分注释详细解释了 BM25 的原理、特性、与密集检索的对比以及常见用途。我们在教程开头已充分讨论。

### 5. 创建 BM25 索引
这是脚本的核心部分：

#### 5.1. 初始化 `BM25Ingestor`
```python
    print("\nInitializing BM25Ingestor and creating the BM25 index...")
    try:
        ingestor = BM25Ingestor()
        print("BM25Ingestor initialized.")
    # ... (初始化错误处理) ...
```
- `ingestor = BM25Ingestor()`: 创建 `BM25Ingestor` 类的实例。这个类的构造函数中可能包含了初始化 BM25 模型（例如，`rank_bm25` 库中的 `BM25Okapi` 类）的逻辑。

#### 5.2. “训练”或“拟合”BM25 模型到语料库
```python
        print(f"Creating BM25 index from {len(all_chunk_texts)} text chunks...")
        bm25_index = ingestor.create_bm25_index(all_chunk_texts)

        if not bm25_index: # 基本检查
            # ... (BM25 索引创建失败处理) ...
            return
        print(f"BM25 index created successfully.")
```
- `bm25_index = ingestor.create_bm25_index(all_chunk_texts)`: **核心步骤**！
    - 这个方法接收包含所有文本块原文的列表 `all_chunk_texts`。
    - 内部实现（例如，使用 `rank_bm25`）大致如下：
        1.  **分词 (Tokenization)**: 将每个文本块分解成词语（tokens）。这通常涉及到去除标点符号、转换为小写等预处理。`rank_bm25` 库的 `BM25Okapi` 类在接收文本列表时会自动进行简单的空格分词。更高级的实现可能会集成更复杂的分词器（如针对特定语言的）。
        2.  **计算统计信息**: 基于分词后的语料库，BM25 算法会计算：
            -   每个词在每个文档中的频率 (Term Frequency, TF)。
            -   每个词的逆文档频率 (Inverse Document Frequency, IDF)，即包含该词的文档数量的倒数（经过平滑处理）。IDF 反映了一个词在整个语料库中的“稀有”或“普遍”程度。越稀有的词，权重通常越高。
        3.  **模型构建**: `bm25_index` 对象（例如一个 `BM25Okapi` 实例）会存储这些计算好的统计信息（如 IDF 值）。它本身就构成了可以用于后续检索的“索引”或“模型”。
- **注意**: 与 FAISS 不同，BM25 索引对象（如 `BM25Okapi` 实例）通常不直接暴露一个像 `.ntotal` 这样的属性来显示其“大小”。它的大小是隐含在其处理过的语料库（`all_chunk_texts`）中的。

#### 5.3. 保存 BM25 索引对象到磁盘
```python
        bm25_output_dir.mkdir(parents=True, exist_ok=True) # 确保输出目录存在
        print(f"Saving BM25 index to: {bm25_index_path}...")
        with open(bm25_index_path, 'wb') as f:
            pickle.dump(bm25_index, f) # 使用 pickle 序列化对象
        print(f"BM25 index successfully saved to {bm25_index_path}")
```
- `pickle.dump(bm25_index, f)`:
    - BM25 模型/索引（如 `BM25Okapi` 对象）是一个包含了词汇表、IDF 值等内部状态的 Python 对象。
    - `pickle` 是 Python 的标准序列化库，可以将几乎任何 Python 对象转换成一个字节流，并保存到文件中。
    - `'wb'` 表示以二进制写模式打开文件，因为 pickle 生成的是字节流。
- 保存后的 `.bm25.pkl` 文件包含了训练好的 BM25 模型，之后可以被加载回来直接用于文档检索，无需重新处理原始文本。

## BM25 索引文件里有什么？

`.bm25.pkl` 文件（通过 pickle 序列化）存储了整个 BM25 模型对象（例如 `rank_bm25.BM25Okapi` 的一个实例）。这个对象内部包含了：
-   根据输入文本块（corpus）计算出的每个词的 IDF（逆文档频率）值。
-   文档的平均长度等统计信息。
-   原始文档列表的一个副本或其处理后的版本（取决于具体库的实现，`rank_bm25` 会存储分词后的文档）。

这个对象在加载后，就可以使用其提供的方法（如 `get_scores(query_tokens)` 或 `get_top_n(query_tokens, documents, n=5)`) 来计算新查询与原始文档集合中每个文档的相关性得分，并返回排序后的结果。

## 如何使用 BM25 索引进行检索？

虽然本 demo 只展示了创建和保存索引，但使用它的大致流程如下：
1.  **加载索引**: 使用 `pickle.load()` 从 `.bm25.pkl` 文件中加载 BM25 对象。
    ```python
    # import pickle
    # with open(bm25_index_path, 'rb') as f:
    #     bm25_retriever = pickle.load(f)
    # # 还需要加载原始的 all_chunk_texts 列表，或者一个 chunk_id 到文本的映射
    # # all_chunk_texts = ... 
    ```
2.  **准备查询**: 对用户的查询字符串进行与索引构建时相同的分词处理。
    ```python
    # user_query = "some search terms"
    # tokenized_query = user_query.lower().split() # 示例简单分词
    ```
3.  **执行检索**: 使用 BM25 对象的方法获取相关文档（文本块）。
    ```python
    # # 获取所有文档的得分
    # scores = bm25_retriever.get_scores(tokenized_query) 
    # # 或者直接获取得分最高的 N 个文档原文 (rank_bm25 需要原始文档列表)
    # top_n_chunks = bm25_retriever.get_top_n(tokenized_query, all_chunk_texts, n=5)
    ```
    这里的 `all_chunk_texts` 就是构建索引时使用的那个文本列表。BM25 返回的是这些原始文本。

## 如何运行脚本

1.  **确保 `demo_07` 已成功运行**: `study/chunked_reports_output/report_for_serialization.json` 文件必须存在，并且包含所有文本块。
2.  **确保 `rank_bm25` 库已安装**: 如果 `BM25Ingestor` 依赖它，你需要 `pip install rank_bm25`。
3.  **打开终端或命令行工具**。
4.  **导航到脚本所在的目录**。
5.  **执行脚本**:
    ```bash
    python study/demo_10_creating_bm25_index.py
    ```
    或者，如果你在项目的根目录下：
    ```bash
    python -m study.demo_10_creating_bm25_index
    ```
    脚本执行后，会在 `study/bm25_indices/` 目录下找到一个 `report_for_serialization.bm25.pkl` 文件。

## 总结：为 RAG 增添关键词检索能力

`demo_10_creating_bm25_index.py` 为我们展示了如何为文本块构建 BM25 索引，从而为 RAG 系统引入了强大的**关键词检索（稀疏检索）**能力。与 `demo_09` 中基于嵌入的 FAISS 索引（密集检索/语义检索）不同，BM25 侧重于词频和精确匹配，两者可以形成有效的互补。

在先进的 RAG 系统中，往往会同时使用这两种检索方式（甚至更多），然后将它们的结果进行智能融合（如 RRF 混合），以期达到比单一检索方法更优的综合效果。

至此，我们关于 PDF 文档解析、处理及检索准备的系列教程就告一段落了。希望这些 Demo 能帮助你理解构建复杂文档智能应用所涉及的关键步骤和技术！
