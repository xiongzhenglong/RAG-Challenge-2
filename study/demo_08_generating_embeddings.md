# 让机器理解文本：`demo_08_generating_embeddings.py` 之生成文本嵌入向量

大家好！在我们的 PDF 处理系列教程中，我们已经成功地将原始 PDF 文档转换、解析、提取，并通过 `demo_07` 将其内容切分成了适合进一步处理的小文本块（chunks）。现在，是时候让机器“理解”这些文本块的含义了。这就要用到一个核心的 AI 概念——**文本嵌入（Text Embeddings）**。

本篇教程将通过 `study/demo_08_generating_embeddings.py` 脚本，向大家展示如何为这些文本块生成向量嵌入。这个过程是构建检索增强生成（RAG）系统的核心步骤之一。我们将使用 `src.ingestion.VectorDBIngestor` 类来辅助完成这个任务，它内部通常会调用预训练的嵌入模型服务（如 OpenAI 的 API）。

## 脚本目标

- 演示如何为从 `demo_07` 中获得的文本块生成文本嵌入向量。
- 解释什么是文本嵌入及其在 RAG 系统中的重要性。
- 强调生成高质量嵌入通常需要 API 支持（例如 OpenAI API Key）。
- 展示嵌入向量的基本特征（如维度）。

## 什么是文本嵌入（Text Embeddings）？

简单来说，**文本嵌入是将文本（单词、短语或整个文本块）转换为一串数字（即一个向量）的过程，这个向量能够捕捉文本的语义含义。**

- **语义相似性**: 含义相近的文本，其对应的嵌入向量在多维向量空间中的位置也更接近。
- **机器可处理**: 计算机不直接理解文本，但它们非常擅长处理数字和向量。文本嵌入将自然语言转换成了机器可以进行计算和比较的格式。

**在 RAG 系统中的重要性：**

1.  **语义搜索 (Semantic Search)**:
    -   当用户提出一个问题（查询 query）时，RAG 系统首先会将这个问题也转换成一个嵌入向量（查询嵌入）。
    -   然后，系统会将这个查询嵌入与预先计算并存储好的所有文本块的嵌入向量进行比较。
2.  **相似度匹配**:
    -   通过计算向量间的相似度（例如使用余弦相似度或点积），系统可以找出与用户查询最相关的那些文本块。
    -   这些最相关的文本块随后会与用户的原始问题一起被提供给大型语言模型（LLM），LLM 则基于这些信息生成一个内容丰富且准确的答案。

**API 需求：**
生成高质量的文本嵌入通常依赖于强大的预训练模型。这些模型一般通过 API 服务提供，例如：
-   OpenAI (如 `text-embedding-ada-002` 模型)
-   Cohere
-   Google (Vertex AI / PaLM API)
这意味着你需要相应服务的 API 密钥才能使用它们。对于 OpenAI，你需要设置 `OPENAI_API_KEY` 环境变量（具体设置方法可见 `study/demo_01_project_setup.py` 教程）。

## 前提条件

1.  **来自 `demo_07` 的切分后报告**: 你需要先成功运行 `study/demo_07_text_splitting.py`。该脚本会在 `study/chunked_reports_output/` 目录下生成一个 JSON 文件（例如 `report_for_serialization.json`），其中包含了切分好的文本块列表 (`content['chunks']`)。
2.  **`OPENAI_API_KEY` 环境变量**: **必须设置**此环境变量，因为 `VectorDBIngestor`（在本 demo 中用于生成嵌入）通常会依赖它来调用 OpenAI 的嵌入模型。

## Python 脚本 `study/demo_08_generating_embeddings.py`

让我们完整地看一下这个脚本的代码：
```python
# study/demo_08_generating_embeddings.py

import json
import os
from pathlib import Path
import sys

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import VectorDBIngestor # VectorDBIngestor handles embedding generation

def main():
    """
    Demonstrates generating text embeddings for selected chunks from a processed report.
    This is a core step in preparing data for Retrieval Augmented Generation (RAG).
    """
    print("Starting text embedding generation demo...")

    # --- 1. Define Paths ---
    # Input is the chunked report (output of demo_07)
    input_chunked_report_dir = Path("study/chunked_reports_output/")
    input_chunked_filename = "report_for_serialization.json" # Assuming this name
    input_chunked_full_path = input_chunked_report_dir / input_chunked_filename

    print(f"Input chunked report directory: {input_chunked_report_dir}")
    print(f"Expected chunked JSON file: {input_chunked_full_path}")

    # --- 2. Prepare Input Data (Load Chunked JSON) ---
    if not input_chunked_full_path.exists():
        print(f"Error: Input chunked JSON file not found at {input_chunked_full_path}")
        print("Please ensure 'demo_07_text_splitting.py' has run successfully.")
        return

    sample_chunks_data = []
    try:
        with open(input_chunked_full_path, 'r', encoding='utf-8') as f:
            chunked_data = json.load(f)
        
        if 'content' in chunked_data and 'chunks' in chunked_data['content'] and chunked_data['content']['chunks']:
            # Extract the first 2-3 chunks for demonstration
            sample_chunks_data = chunked_data['content']['chunks'][:3]
            if not sample_chunks_data:
                 print("No chunks found in the loaded JSON file.")
                 return
            print(f"Successfully loaded chunked JSON. Using {len(sample_chunks_data)} sample chunks for demo.")
        else:
            print("Error: 'content' or 'chunks' not found in the loaded JSON structure.")
            print("Please ensure the input file is correctly formatted (output of demo_07).")
            return
            
    except json.JSONDecodeError:
        print(f"Error: Could not decode the JSON file at {input_chunked_full_path}.")
        return
    except Exception as e:
        print(f"An error occurred while loading the JSON file: {e}")
        return

    # --- 3. Understanding Text Embeddings ---
    # Text embeddings are numerical representations (vectors) of text that capture
    # its semantic meaning. Words, phrases, or entire text chunks with similar meanings
    # will have embeddings that are close together in the vector space.
    #
    # Importance in RAG:
    #   - Semantic Search: When a user asks a question (query), the query is also
    #     converted into an embedding. The RAG system then compares this query
    #     embedding with the embeddings of all stored text chunks.
    #   - Similarity Matching: Chunks whose embeddings are most similar (e.g., by
    #     cosine similarity or dot product) to the query embedding are considered
    #     the most relevant. These relevant chunks are then provided to an LLM
    #     along with the original query to generate an informed answer.
    #
    # API Requirement:
    #   - Generating high-quality embeddings typically requires using pre-trained models
    #     provided by services like OpenAI (e.g., 'text-embedding-ada-002'),
    #     Cohere, Google, etc.
    #   - This usually involves making API calls, which means an API key for the chosen
    #     service must be configured in your environment. For OpenAI, this is the
    #     `OPENAI_API_KEY`. Refer to `study/demo_01_project_setup.py` for how to set this up.

    # --- 4. Generate Embeddings for Sample Chunks ---
    print("\nInitializing VectorDBIngestor to generate embeddings...")
    
    # Check for API key before attempting to initialize or make calls
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Embeddings generation requires an OpenAI API key.")
        print("Please set it up (see demo_01_project_setup.py) and try again.")
        return

    try:
        # VectorDBIngestor sets up the LLM client (e.g., OpenAI) based on environment variables
        # or configuration. It provides methods for embedding generation.
        ingestor = VectorDBIngestor() 
        print("VectorDBIngestor initialized successfully.")
    except Exception as e:
        print(f"Error initializing VectorDBIngestor: {e}")
        print("This might be due to missing API keys or other configuration issues.")
        return

    print("\n--- Generating Embeddings for Sample Chunks ---")
    print("(This involves API calls to an embedding model, e.g., OpenAI's ada-002)")

    for i, chunk_data in enumerate(sample_chunks_data):
        chunk_text = chunk_data.get('text', '')
        chunk_id = chunk_data.get('id', f'sample_chunk_{i+1}')

        if not chunk_text:
            print(f"\nSkipping chunk {chunk_id} as it has no text content.")
            continue

        print(f"\n--- Chunk ID: {chunk_id} ---")
        print(f"  Text (Snippet): \"{chunk_text[:150]}...\"")

        try:
            # `_get_embeddings` expects a list of texts and returns a list of embeddings.
            # For a single chunk, we pass a list containing its text.
            embedding_list = ingestor._get_embeddings([chunk_text])
            
            if embedding_list and isinstance(embedding_list, list) and len(embedding_list) > 0:
                embedding_vector = embedding_list[0] # Get the first (and only) embedding
                
                print(f"  Embedding Generated Successfully.")
                # Most OpenAI embeddings (like ada-002) have 1536 dimensions.
                print(f"  Total Dimensionality: {len(embedding_vector)}")
                # Print the first few dimensions to give an idea of the vector
                print(f"  First 10 Dimensions (Sample): {embedding_vector[:10]}")
            else:
                print("  Error: _get_embeddings did not return the expected list of embeddings.")

        except Exception as e:
            print(f"  Error generating embedding for chunk {chunk_id}: {e}")
            print("  This could be due to API issues (rate limits, key problems), network problems,")
            print("  or issues with the input text format/length for the embedding model.")
            # Optionally, break or continue based on error handling strategy
            # For this demo, we'll continue to try other chunks.
    print("---------------------------------------------")

    print("\nEmbedding generation demo complete.")
    print("Note: Actual storage of these embeddings into a vector database is covered")
    print("in the next steps of a full RAG pipeline (e.g., using VectorDBIngestor.ingest_reports).")

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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import VectorDBIngestor # 核心：处理嵌入生成和向量数据库交互
```
- `VectorDBIngestor`: 这个类名暗示了它最终的目的是将数据“摄入”（ingest）到向量数据库中。然而，在摄入之前，一个关键步骤就是为文本数据生成嵌入向量。因此，这个类也封装了与嵌入模型API交互的逻辑。

### 2. 定义路径
```python
    input_chunked_report_dir = Path("study/chunked_reports_output/")
    input_chunked_filename = "report_for_serialization.json" # 与 demo_07 输出对应
    input_chunked_full_path = input_chunked_report_dir / input_chunked_filename
```
- `input_chunked_full_path`: 指向 `demo_07` 生成的、包含文本块列表（`content['chunks']`）的 JSON 文件。

### 3. 准备输入数据（加载切分后的 JSON）
```python
    if not input_chunked_full_path.exists():
        # ... 错误处理: 输入文件不存在 ...
        return

    sample_chunks_data = []
    try:
        with open(input_chunked_full_path, 'r', encoding='utf-8') as f:
            chunked_data = json.load(f)
        
        if 'content' in chunked_data and 'chunks' in chunked_data['content'] and chunked_data['content']['chunks']:
            sample_chunks_data = chunked_data['content']['chunks'][:3] # 取前3个块作为示例
            # ... (检查 sample_chunks_data 是否为空) ...
        else:
            # ... 错误处理: JSON 结构不符合预期 ...
            return
    # ... (JSON 解码等错误处理) ...
```
- 脚本首先加载 `demo_07` 的输出文件。
- **选取样本数据**: 为了避免在演示中进行过多（可能产生费用且耗时）的 API 调用，脚本从加载的文本块列表中只选取了前 `3` 个块 (`[:3]`) 作为 `sample_chunks_data` 进行后续的嵌入生成演示。在实际应用中，你需要处理所有的文本块。

### 4. 理解文本嵌入（脚本中的第 3 部分注释）
这部分注释详细解释了文本嵌入的定义、其在 RAG 中的核心作用（语义搜索、相似度匹配）以及对 API 密钥（如 `OPENAI_API_KEY`）的依赖。我们在教程开头已经对此进行了充分的阐述。

### 5. 为样本块生成嵌入向量
```python
    print("\nInitializing VectorDBIngestor to generate embeddings...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        # ... (提示并退出) ...
        return

    try:
        ingestor = VectorDBIngestor() 
        print("VectorDBIngestor initialized successfully.")
    except Exception as e:
        # ... 初始化错误处理 ...
        return

    print("\n--- Generating Embeddings for Sample Chunks ---")
    print("(This involves API calls to an embedding model, e.g., OpenAI's ada-002)")

    for i, chunk_data in enumerate(sample_chunks_data):
        chunk_text = chunk_data.get('text', '')
        chunk_id = chunk_data.get('id', f'sample_chunk_{i+1}')

        if not chunk_text: # 跳过空文本块
            continue

        print(f"\n--- Chunk ID: {chunk_id} ---")
        print(f"  Text (Snippet): \"{chunk_text[:150]}...\"")

        try:
            embedding_list = ingestor._get_embeddings([chunk_text]) # 核心调用
            
            if embedding_list and isinstance(embedding_list, list) and len(embedding_list) > 0:
                embedding_vector = embedding_list[0]
                
                print(f"  Embedding Generated Successfully.")
                print(f"  Total Dimensionality: {len(embedding_vector)}")
                print(f"  First 10 Dimensions (Sample): {embedding_vector[:10]}")
            else:
                # ... _get_embeddings 返回结果不符合预期的错误处理 ...
        except Exception as e:
            # ... 生成嵌入过程中的 API 或网络错误处理 ...
    print("---------------------------------------------")
```
- **API 密钥检查**: **再次强调**，`os.getenv("OPENAI_API_KEY")` 用于检查 OpenAI API 密钥是否已设置。没有它，后续步骤将失败。
- **初始化 `VectorDBIngestor`**:
    - `ingestor = VectorDBIngestor()`: 创建 `VectorDBIngestor` 类的实例。这个类的构造函数中很可能包含了初始化与嵌入模型服务（如 OpenAI）客户端的逻辑，它会读取环境变量中的 API 密钥。
- **遍历样本块并生成嵌入**:
    - 脚本遍历之前选取的 `sample_chunks_data`。
    - `chunk_text = chunk_data.get('text', '')`: 获取每个块的实际文本内容。
    - `embedding_list = ingestor._get_embeddings([chunk_text])`:
        - 这是**生成嵌入的核心调用**。虽然方法名前有一个下划线 `_`（通常表示内部使用），但在这个 demo 中它被用来直接获取嵌入。
        - 该方法接收一个**文本列表**作为输入（即使只有一个文本块，也需要将其放入列表中，如 `[chunk_text]`）。
        - 它返回一个**嵌入向量的列表**。由于我们每次只传入一个文本块，所以返回的列表也只包含一个嵌入向量。
    - `embedding_vector = embedding_list[0]`: 提取出该文本块对应的嵌入向量。
- **显示嵌入信息**:
    - `Total Dimensionality: {len(embedding_vector)}`: 打印嵌入向量的维度。例如，OpenAI 的 `text-embedding-ada-002` 模型生成的嵌入向量是 **1536** 维的。
    - `First 10 Dimensions (Sample): {embedding_vector[:10]}`: 打印向量的前 10 个数字，让我们对这个高维向量有一个直观的（尽管非常局部）感受。
- **错误处理**: 脚本包含了对 `VectorDBIngestor` 初始化失败和单个块嵌入生成失败的错误处理。

### 6. 提示后续步骤
脚本最后打印提示信息，说明实际将这些嵌入向量存入向量数据库（Vector Database）是构建完整 RAG 流水线的后续步骤（例如，可能会调用 `VectorDBIngestor.ingest_reports` 这样的方法）。本 demo 只聚焦于“生成”嵌入这一环节。

## 下一步是什么？向量数据库！

生成的这些嵌入向量（连同它们对应的原始文本块 ID、文本内容和元数据）的最终归宿通常是**向量数据库**（Vector Database），例如 FAISS, Pinecone, Weaviate, Milvus, ChromaDB 等。

向量数据库专门为高效存储、索引和查询大规模高维向量数据而设计。当用户提问时，其问题也会被转换为嵌入向量，然后在向量数据库中执行快速的相似性搜索，以找到最相关的文本块嵌入，进而取回这些文本块用于 LLM 生成答案。

## 如何运行脚本

1.  **确保 `demo_07` 已成功运行**: `study/chunked_reports_output/report_for_serialization.json` 文件必须存在。
2.  **设置 `OPENAI_API_KEY` 环境变量**: **至关重要！**
    ```bash
    export OPENAI_API_KEY="sk-YourActualOpenAIKeyHere" 
    ```
    (在 Windows CMD 中使用 `set OPENAI_API_KEY=sk-YourActualOpenAIKeyHere`)
3.  **打开终端或命令行工具**。
4.  **导航到脚本所在的目录**。
5.  **执行脚本**:
    ```bash
    python study/demo_08_generating_embeddings.py
    ```
    或者，如果你在项目的根目录下：
    ```bash
    python -m study.demo_08_generating_embeddings
    ```
    脚本会为示例的几个文本块调用 OpenAI API 来生成嵌入。你会看到每个块的文本片段、其嵌入向量的维度以及前几个维度的值。

## 总结

`demo_08_generating_embeddings.py` 向我们展示了 RAG 流程中一个非常核心的技术环节：**文本嵌入生成**。通过将文本块转换为能够捕捉其语义的数字向量，我们为机器实现基于含义的文本理解和检索奠定了基础。这些嵌入向量是后续进行语义搜索、相似性匹配以及最终由 LLM 生成高质量答案的关键。

虽然本 demo 只处理了少量样本数据，但在实际应用中，你需要为文档库中的所有文本块生成并存储嵌入。理解了这一步，你就离构建一个强大的 RAG 应用更近了！希望这篇教程对你有所帮助！
