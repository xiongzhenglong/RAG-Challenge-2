# RAG 结果优化：`demo_16_llm_reranking.py` 之 LLM 重排序（Reranking）

大家好！在我们的检索增强生成（RAG）系列教程中，我们已经学习了如何通过 FAISS (`demo_11`) 进行语义检索和通过 BM25 (`demo_12`) 进行关键词检索来获取与用户问题相关的文本块。这些初始检索方法（也称为一级检索或召回）能帮我们从大量文档中快速筛选出一批候选结果。但是，这些初步结果的排序未必总是最优，最相关的文档可能没有排在最前面。

为了进一步提升最终答案的质量，我们可以引入一个“重排序（Reranking）”步骤。本篇教程将通过 `study/demo_16_llm_reranking.py` 脚本，向大家展示如何利用大型语言模型（LLM）的强大理解能力，对初步检索到的文本块进行更精细的评估和重新排序。我们将使用 `src.reranking.LLMReranker` 类来完成这个任务。

## 脚本目标

- 演示如何使用 `LLMReranker` 类对一组初步检索到的文档（文本块）进行基于 LLM 的重排序。
- 理解 LLM 在重排序过程中的作用：评估每个候选文档与用户查询的精确相关性。
- 解释如何结合初始检索得分和 LLM 评估得分，生成一个最终的“组合得分”（combined_score）并据此排序。
- 强调此过程对于提升 RAG 系统最终输出结果的精准度的重要性。

## 什么是 LLM 重排序？

LLM 重排序是在初始检索（召回）之后的一个**第二阶段优化步骤**。其核心思想是：

1.  **获取候选文档**: 首先，通过一种或多种初步检索方法（如 BM25、FAISS 向量搜索，或两者的某种组合）从知识库中获取一个候选文档列表（例如，Top 20 或 Top 50 的结果）。
2.  **LLM 精细评估**: 接着，利用一个强大的大型语言模型（LLM，如 GPT-4o-mini, GPT-4 等）来**逐个或分批次地**评估这些候选文档中的每一个与原始用户查询的**具体相关性**。
    -   LLM 会被提供用户查询和单个候选文档（或一小批文档）的文本内容。
    -   通过精心设计的 Prompt（例如 `demo_13` 中讨论的 `RerankingPrompt`），LLM 被要求判断该文档是否与查询相关，并给出一个**相关性分数 (relevance_score)**，有时甚至会给出判断的**理由 (reasoning)**。这个输出通常是结构化的（例如，符合 `RetrievalRankingSingleBlock` Pydantic 模型）。
3.  **计算组合得分**: 为了得到最终的排序，系统会将初始检索方法给出的原始得分（例如，向量搜索的距离/相似度，或 BM25 的得分）与 LLM 给出的 `relevance_score` 进行加权组合，形成一个 `combined_score`。
4.  **最终排序**: 所有候选文档根据这个 `combined_score` 进行降序排列，得到一个更精确、更可靠的相关文档列表。

**为什么需要重排序？**
-   **初步检索的局限性**: 向量搜索可能召回语义相关但不够直接的文档；BM25 可能只匹配了关键词但忽略了深层含义。
-   **LLM 的深度理解**: LLM 拥有更强的自然语言理解能力，能更准确地判断细微的语义差别和上下文关联，从而识别出那些对回答用户问题真正有价值的信息。
-   **成本与效率的平衡**: 直接让 LLM 处理所有文档不现实（成本高、速度慢）。重排序只针对初步筛选出的一小部分候选文档进行，是计算成本和结果质量之间的一个有效折衷。

## 前提条件

1.  **`OPENAI_API_KEY` 环境变量**: **至关重要！** `LLMReranker` 需要调用 OpenAI API 来执行评估，因此必须正确设置此密钥。
2.  **示例数据**: 本 demo 为了简化，直接在脚本中定义了 `sample_query`（示例用户查询）和 `sample_documents`（一个模拟的初步检索结果列表，每个文档包含文本、ID、初始得分等）。在实际应用中，`sample_documents` 会来自 `demo_11` 或 `demo_12` 的输出。

## Python 脚本 `study/demo_16_llm_reranking.py`

让我们完整地看一下这个脚本的代码：
```python
# study/demo_16_llm_reranking.py

import sys
import os
import json
from dotenv import load_dotenv

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.reranking import LLMReranker

# Load environment variables from .env file (especially OPENAI_API_KEY)
load_dotenv()

# --- Purpose of this Demo ---
# This script demonstrates LLM-based reranking of search results.
# Initial retrieval methods (like vector search or BM25) provide a set of candidate
# documents. LLM-based reranking uses a more powerful Language Model to assess
# the relevance of these candidates to the query in a more nuanced way.
# The LLM typically assigns a relevance score, which can then be combined with
# the initial retrieval score to produce a final, refined ranking.
# This helps to improve the precision of the top results shown to the user or
# passed to a final answer generation step.
#
# IMPORTANT:
# An `OPENAI_API_KEY` must be set in your .env file in the project root,
# as `LLMReranker` uses an LLM (e.g., OpenAI's GPT models) to perform the reranking.

def main():
    """
    Demonstrates LLM-based reranking of a sample list of documents
    against a sample query.
    """
    print("Starting LLM-based reranking demo...")

    # --- 1. Prepare Sample Data ---
    sample_query = "Tell me about the company's sustainability efforts and impact on local communities."

    sample_documents = [
        {
            "id": "doc1",
            "text": "Our company is committed to reducing its carbon footprint. We have invested in renewable energy sources and aim for carbon neutrality by 2030. Our annual report details these initiatives.",
            "distance": 0.85, # Initial similarity score (higher is better)
            "page": 5
        },
        {
            "id": "doc2",
            "text": "The new product line, launched in Q3, has exceeded sales expectations. Marketing strategies focused on digital channels and influencer collaborations.",
            "distance": 0.72,
            "page": 12
        },
        {
            "id": "doc3",
            "text": "We actively engage with local communities through various programs, including educational scholarships and environmental cleanup drives. Last year, we dedicated over 1000 volunteer hours to these causes.",
            "distance": 0.90,
            "page": 8
        },
        {
            "id": "doc4",
            "text": "Financial highlights for the fiscal year include a 15% increase in revenue. The primary drivers were strong performance in the North American market and cost optimization measures.",
            "distance": 0.65,
            "page": 2
        }
    ]

    print("\n--- Sample Query ---")
    print(f"  \"{sample_query}\"")

    print("\n--- Documents Before Reranking (Sorted by initial 'distance' score desc) ---")
    # Sort by initial distance for clarity before reranking
    sorted_initial_docs = sorted(sample_documents, key=lambda x: x['distance'], reverse=True)
    for doc in sorted_initial_docs:
        print(f"  ID: {doc['id']}, Initial Distance (Similarity): {doc['distance']:.2f}, "
              f"Page: {doc.get('page', 'N/A')}, Text Snippet: \"{doc['text'][:100]}...\"")

    # --- 2. Check for API Key ---
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("\nError: OPENAI_API_KEY environment variable is not set.")
        print("Please ensure your API key is configured in the .env file in the project root.")
        return
    print("\nOPENAI_API_KEY found in environment.")

    # --- 3. Initialize LLMReranker ---
    try:
        # LLMReranker uses an LLM (defaulting to OpenAI's models)
        # to assess the relevance of each document to the query.
        reranker = LLMReranker()
        print("LLMReranker initialized.")
    except Exception as e:
        print(f"Error initializing LLMReranker: {e}")
        return

    # --- 4. Perform Reranking ---
    print("\nPerforming reranking using LLM...")
    print("(This may take some time as it involves LLM calls for document batches)...")

    try:
        # `rerank_documents` processes documents in batches, sends them to the LLM
        # with the query, and gets relevance scores.
        # `llm_weight` controls how much the LLM's relevance score influences the
        # final combined score relative to the initial retrieval score ('distance').
        reranked_documents = reranker.rerank_documents(
            query=sample_query,
            documents=sample_documents,
            documents_batch_size=2, # Process 2 documents per LLM call for this demo
            llm_weight=0.7          # Give LLM's assessment more weight
        )

        print("\n--- Documents After Reranking (Sorted by 'combined_score' desc) ---")
        if not reranked_documents:
            print("  No documents returned after reranking.")
        else:
            # The reranked_documents list should already be sorted by 'combined_score'
            for doc in reranked_documents:
                print(f"  ID: {doc['id']}")
                print(f"    Original Distance (Similarity): {doc.get('distance', 'N/A'):.2f}")
                print(f"    LLM Relevance Score: {doc.get('relevance_score', 'N/A'):.2f}")
                print(f"    Combined Score: {doc.get('combined_score', 'N/A'):.2f}")
                print(f"    Page: {doc.get('page', 'N/A')}")
                print(f"    Text Snippet: \"{doc.get('text', '')[:100]}...\"")
                print(f"    LLM Reasoning (if available): {doc.get('reasoning', 'N/A')}")
                print("-" * 20)
            
            print("\n  Note on Ordering:")
            print("  The order of documents may have changed significantly from the initial ranking.")
            print("  Documents with high LLM relevance scores can be promoted even if their initial")
            print("  'distance' (similarity) score was not the highest, and vice-versa.")
            print("  The 'combined_score' reflects this new, refined ranking.")

    except Exception as e:
        print(f"\nAn error occurred during the reranking process: {e}")
        print("This could be due to API key issues, network problems, LLM model errors,")
        print("or issues with the input data format.")
        import traceback
        traceback.print_exc()

    print("\nLLM-based reranking demo complete.")

if __name__ == "__main__":
    main()
```

## 脚本代码详解

### 1. 导入模块
```python
import sys
import os
import json # 虽然本脚本未直接使用，但 LLM 交互常涉及 JSON
from dotenv import load_dotenv # 用于加载 .env 文件

sys.path.append(...) # 添加 src 目录

from src.reranking import LLMReranker # 核心：LLM 重排序器

load_dotenv() # 加载环境变量
```
- `LLMReranker`: 这是我们自己封装的类（在 `src.reranking.py` 中定义），它负责协调与 LLM 的交互以完成重排序任务。它内部可能会使用 `APIProcessor` 以及 `demo_13` 中讨论的 `RerankingPrompt` 和相关的 Pydantic 模型。

### 2. `main()` 函数

#### 2.1. 准备示例数据
```python
    sample_query = "Tell me about the company's sustainability efforts and impact on local communities."
    sample_documents = [
        {
            "id": "doc1",
            "text": "Our company is committed to reducing its carbon footprint...",
            "distance": 0.85, # 初始检索得分（越高越好，假设是相似度）
            "page": 5
        },
        # ... 其他示例文档 ...
    ]
    # ... 打印重排序前的文档列表 (按初始得分排序) ...
```
- `sample_query`: 一个代表用户提出的、略带复杂性的查询。
- `sample_documents`: 这是一个 Python 列表，其中每个元素是一个字典，代表一个**已经被初步检索系统（如 FAISS 或 BM25）找回的文档（文本块）**。
    - `id`: 文档的唯一标识。
    - `text`: 文档的实际文本内容。
    - `distance`: **初始检索得分**。这个值可能来自向量搜索的相似度（此时通常越高越好），也可能来自 BM25 的得分（也是越高越好），或者是某种距离度量（此时可能越低越好，但 `LLMReranker` 内部会做适配或期望一种统一的“越高越好”的输入）。在这个 demo 中，注释明确指出“Initial similarity score (higher is better)”，所以我们假设它是一个相似度得分。
    - `page`: （可选）文档来源的页码。
- 为了对比，脚本首先会按照 `distance`（初始得分）对这些文档进行排序并打印。

#### 2.2. 检查 API 密钥
与之前的 API 调用 demo 类似，脚本会检查 `OPENAI_API_KEY` 是否已设置。

#### 2.3. 初始化 `LLMReranker`
```python
    try:
        reranker = LLMReranker()
        print("LLMReranker initialized.")
    except Exception as e:
        # ... 初始化错误处理 ...
        return
```
- `reranker = LLMReranker()`: 创建 `LLMReranker` 实例。这个类的构造函数可能会初始化其内部使用的 `APIProcessor` 或直接的 LLM 客户端。

#### 2.4. 执行重排序
```python
    print("\nPerforming reranking using LLM...")
    print("(This may take some time as it involves LLM calls for document batches)...")
    try:
        reranked_documents = reranker.rerank_documents(
            query=sample_query,
            documents=sample_documents,
            documents_batch_size=2, # 每次 LLM 调用处理2个文档
            llm_weight=0.7          # LLM 评估的权重占 70%
        )
```
- `reranker.rerank_documents(...)`: 这是执行重排序的核心方法。参数包括：
    - `query`: 用户的原始查询。
    - `documents`: 包含初步检索结果的 `sample_documents` 列表。
    - `documents_batch_size=2`: **批处理大小**。为了优化对 LLM API 的调用（例如，避免超出单个请求的 token 限制，或减少请求次数），`LLMReranker` 可能将文档分批发送给 LLM 进行评估。这里设置为2，表示每次调用 LLM API 时，会同时发送用户查询和2个文档的文本内容，让 LLM 对这2个文档进行相关性评估。
    - `llm_weight=0.7`: **LLM 评估权重**。这是一个非常关键的参数，它决定了在计算最终的“组合得分”(`combined_score`)时，LLM 给出的相关性评估 (`relevance_score`) 占多大比重，而原始的检索得分 (`distance`) 占多大比重。
        - `combined_score = llm_weight * llm_relevance_score + (1 - llm_weight) * initial_retrieval_score` (这是一个可能的计算公式，具体实现可能有所不同，但概念一致)。
        - `llm_weight=0.7`意味着最终排序更侧重于 LLM 的判断（70%），而原始检索得分的贡献占 30%。这个权重需要根据实际效果进行调整。

- **`rerank_documents` 内部大致流程**:
    1.  遍历 `documents` 列表（可能按 `documents_batch_size` 分批）。
    2.  对于每个文档（或一批文档），构造一个特定的 Prompt（例如，使用 `src.prompts.RerankingPrompt`）。这个 Prompt 会包含用户查询和当前文档（或这批文档）的文本。
    3.  调用 LLM API（通过内部的 `APIProcessor`）发送这个 Prompt。
    4.  期望 LLM 返回一个结构化的 JSON 响应（例如，符合 `src.prompts.RetrievalRankingSingleBlock` 或 `RetrievalRankingMultipleBlocks` Pydantic 模型），其中包含对该文档（或该批次中每个文档）的 `relevance_score`（例如，一个0到1之间的浮点数）和可选的 `reasoning`（LLM 给出判断的理由）。
    5.  对于每个文档，根据 `llm_weight`、LLM 返回的 `relevance_score` 以及文档原始的 `distance`（初始检索得分），计算出一个 `combined_score`。
    6.  所有处理过的文档（现在都带有 `relevance_score` 和 `combined_score`）会根据 `combined_score` 进行**降序排列**。
    7.  返回这个重排序后的文档列表。

#### 2.5. 显示重排序后的结果
```python
        print("\n--- Documents After Reranking (Sorted by 'combined_score' desc) ---")
        if not reranked_documents:
            # ...
        else:
            for doc in reranked_documents: # 列表已经是按 combined_score 排序好的
                print(f"  ID: {doc['id']}")
                print(f"    Original Distance (Similarity): {doc.get('distance', 'N/A'):.2f}")
                print(f"    LLM Relevance Score: {doc.get('relevance_score', 'N/A'):.2f}")
                print(f"    Combined Score: {doc.get('combined_score', 'N/A'):.2f}")
                # ... 打印页面、文本片段和 LLM 理由 ...
            # ... 打印排序说明 ...
```
- 脚本遍历 `reranked_documents` 列表（这个列表应该已经是 `LLMReranker` 内部按 `combined_score` 降序排好的）。
- 对于每个文档，它会打印出：
    - `Original Distance (Similarity)`: 初始检索得分。
    - `LLM Relevance Score`: LLM 对该文档与查询相关性的评估得分。
    - `Combined Score`: 结合了上述两者的新总分。
    - `LLM Reasoning` (如果可用): LLM 给出其相关性判断的文字解释。
- **排序变化说明**: 脚本最后强调，重排序后的文档顺序可能与初始排序有显著不同。LLM 的评估可能会将初始得分不高但实际非常相关的文档提升到前面，反之亦然。`combined_score` 体现了这种更精细的排序。

## 关键启示

1.  **重排序是提升检索质量的有效手段**: 它利用 LLM 的深度理解能力来优化初步检索结果。
2.  **结构化 Prompt 和输出**: `LLMReranker` 内部依赖于精心设计的 Prompt（如 `RerankingPrompt`）和 Pydantic 模型来确保与 LLM 的有效沟通和获取可靠的、结构化的评估结果。
3.  **权重调整的重要性**: `llm_weight` 参数的调整对于平衡初始检索和 LLM 重排序的影响至关重要，需要实验来找到最佳值。
4.  **批处理**: `documents_batch_size` 参数展示了在与 LLM 交互时进行批处理以提高效率或管理限制的实际考虑。
5.  **解释性**: LLM 返回的 `reasoning` 可以为用户（或开发者）提供关于为何某个文档被认为是相关或不相关的宝贵见解。

## 如何运行脚本

1.  **设置 `OPENAI_API_KEY`**: 确保在 `.env` 文件中正确配置了你的 OpenAI API 密钥。
2.  **确保相关库已安装**: `pip install openai python-dotenv pydantic` (以及 `LLMReranker` 可能依赖的其他库)。
3.  **打开终端或命令行工具**。
4.  **导航到脚本所在的目录**。
5.  **执行脚本**:
    ```bash
    python study/demo_16_llm_reranking.py
    ```
    或者，如果你在项目的根目录下：
    ```bash
    python -m study.demo_16_llm_reranking
    ```
    脚本将会：
    - 显示示例文档的初始排序。
    - 调用 OpenAI API（例如 `gpt-4o-mini` 模型）对每个（或每批）示例文档进行相关性评估。
    - 计算组合得分。
    - 打印出根据组合得分重排序后的文档列表，包括 LLM 给出的相关性分数和可能的理由。

## 总结

`demo_16_llm_reranking.py` 为我们演示了 RAG 流程中一个高级且非常有效的优化步骤——LLM 重排序。通过让 LLM 对初步检索结果进行更深层次的“二次筛选”，我们可以显著提高最终送往答案生成模块的上下文信息的质量和相关性，从而最终提升整个 RAG 系统的表现。

这虽然会增加一些 API 调用成本和处理时间，但在许多对答案精度要求较高的场景下，这种投入是值得的。希望本教程能帮助你理解并应用 LLM 重排序技术！
