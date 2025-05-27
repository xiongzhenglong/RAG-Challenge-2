# RAG 查询理解进阶：`demo_19_rephrasing_comparative_questions.py` 之比较型问题改写

大家好！在构建高级检索增强生成（RAG）系统的过程中，我们不仅要处理简单的直接问题，还需要应对更复杂的查询类型。其中一种常见且有挑战性的是**比较型问题**，例如“对比A公司和B公司在某个指标上的表现”或“X产品和Y产品各有什么优缺点？”

直接将这类宽泛的比较型问题用于信息检索，效率可能不高，或者难以一次性找到支持完整比较的所有相关信息。一个更有效的策略是**“分而治之”**：首先将原始的比较型问题分解成针对每个比较对象的、更具体、更聚焦的子问题。

本篇教程将通过 `study/demo_19_rephrasing_comparative_questions.py` 脚本，向大家展示如何利用大型语言模型（LLM）的能力，通过 `src.api_requests.APIProcessor` 类中的 `get_rephrased_questions` 方法，将一个涉及多家公司的比较型问题，改写成针对每家公司的独立子问题。

## 脚本目标

- 演示如何针对包含多个实体（例如公司）的比较型问题，生成针对每个实体的、独立的、更易于检索的子问题。
- 理解为何在 RAG 系统中对比较型问题进行改写是一个重要的预处理步骤。
- 展示如何使用 `APIProcessor.get_rephrased_questions` 方法调用 LLM 来完成这个改写任务。
- 查看 LLM 为每个实体生成的具体改写后的问题。

## 为何要改写比较型问题？

对比较型问题进行分解和改写，对于 RAG 系统来说具有显著优势：

1.  **更精准的靶向检索 (Targeted Retrieval)**:
    -   RAG 系统的检索模块（无论是基于向量的还是基于关键词的）通常在处理目标明确、单一的查询时效果最佳。
    -   将“对比A和B的X指标”这样的问题，分解成“A的X指标是什么？”和“B的X指标是什么？”两个独立的子问题后，系统可以分别为A和B执行更精确的检索，找到各自最相关的文档片段。
2.  **更清晰的上下文 (Contextual Clarity)**:
    -   在后续的答案生成阶段，如果LLM分别接收到关于A公司X指标的清晰上下文，和关于B公司X指标的清晰上下文，它能更容易地理解和区分两者，从而为每个实体提取出准确的事实。
3.  **更简化的答案合成 (Simplified Answer Synthesis)**:
    -   当系统为每个实体的每个相关方面都收集到了独立、准确的信息后，最后一步——合成一个全面的比较型答案——就会变得相对容易。这可以由另一个专门的 LLM 调用（例如使用 `demo_13` 中讨论的 `ComparativeAnswerPrompt`）来完成，它接收所有收集到的独立事实，并进行总结和比较。

`APIProcessor.get_rephrased_questions` 方法内部很可能使用了一个类似于 `src.prompts.RephrasedQuestionsPrompt`（我们在 `demo_13` 中见过其通用版本）的 Prompt 结构，但这个 Prompt 会被特别设计来指导 LLM 执行针对多实体的比较型问题分解任务。

## 前提条件

- **`OPENAI_API_KEY` 环境变量**: **至关重要！** 此脚本需要调用 OpenAI API 来执行问题改写，因此必须正确设置此密钥。

## Python 脚本 `study/demo_19_rephrasing_comparative_questions.py`

让我们完整地看一下这个脚本的代码：
```python
# study/demo_19_rephrasing_comparative_questions.py

import sys
import os
import json
from dotenv import load_dotenv

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api_requests import APIProcessor # Used for making LLM calls

# Load environment variables from .env file (especially OPENAI_API_KEY)
load_dotenv()

# --- Purpose of this Demo ---
# This script demonstrates how comparative questions involving multiple entities
# (e.g., companies) are rephrased into individual, focused questions for each entity.
#
# Why rephrase comparative questions for RAG?
#   - Targeted Retrieval: RAG systems are often optimized to retrieve information
#     for a single, clear query. Rephrasing allows the system to fetch the most
#     relevant documents for each entity separately.
#   - Contextual Clarity: When generating an answer, providing the LLM with context
#     specific to "AlphaCorp's R&D" and then separately for "BetaInc's R&D" can lead
#     to more accurate and well-supported individual facts.
#   - Simplified Answer Synthesis: Once individual pieces of information are gathered
#     for each entity, it's often easier for a subsequent LLM call (or even procedural logic)
#     to synthesize a comparative answer.
#
# The `APIProcessor.get_rephrased_questions` method internally uses an LLM
# (prompted by a structure similar to `src.prompts.RephrasedQuestionsPrompt`)
# to generate these entity-specific questions.
#
# IMPORTANT:
# An `OPENAI_API_KEY` must be set in your .env file in the project root.

def main():
    """
    Demonstrates rephrasing a comparative question into individual questions
    for each involved company using APIProcessor.
    """
    print("Starting comparative question rephrasing demo...")

    # --- 1. Check for API Key ---
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please ensure your API key is configured in the .env file in the project root.")
        return
    print("OPENAI_API_KEY found in environment.")

    # --- 2. Initialize APIProcessor ---
    try:
        api_processor = APIProcessor(provider="openai")
        print("APIProcessor initialized for OpenAI.")
    except Exception as e:
        print(f"Error initializing APIProcessor: {e}")
        return

    # --- 3. Define Sample Comparative Question and Companies ---
    sample_comparative_question = (
        "Compare the Research and Development (R&D) expenses of 'AlphaCorp' and 'BetaInc' for the fiscal year 2023. "
        "Which company invested more in R&D during this period?"
    )
    involved_companies = ["AlphaCorp", "BetaInc"]

    print("\n--- Original Comparative Question ---")
    print(f"  Question: \"{sample_comparative_question}\"")
    print(f"  Involved Companies: {involved_companies}")

    # --- 4. Rephrase the Comparative Question ---
    print("\nRequesting rephrased questions from LLM via APIProcessor...")
    print("(This involves an LLM call using a prompt similar to RephrasedQuestionsPrompt)...")

    try:
        # `get_rephrased_questions` uses an LLM to break down the comparative question
        # into individual questions for each company.
        rephrased_map = api_processor.get_rephrased_questions(
            original_question=sample_comparative_question,
            companies=involved_companies
        )

        print("\n--- Rephrased Questions Map ---")
        if rephrased_map and isinstance(rephrased_map, dict):
            for company, rephrased_q in rephrased_map.items():
                print(f"  For Company '{company}':")
                print(f"    Rephrased Question: \"{rephrased_q}\"")
        else:
            print("  Failed to get a valid rephrased questions map or the map was empty.")
            print(f"  Received: {rephrased_map}")

        # --- 5. Inspect API Response Data from the Rephrasing Call ---
        print("\n--- API Response Metadata (from api_processor.processor.response_data for rephrasing call) ---")
        if hasattr(api_processor.processor, 'response_data') and api_processor.processor.response_data:
            response_metadata = api_processor.processor.response_data
            
            if hasattr(response_metadata, 'model'):
                 print(f"  Model Used (from API): {response_metadata.model}")

            if hasattr(response_metadata, 'usage') and response_metadata.usage:
                usage_info = response_metadata.usage
                print(f"  Token Usage:")
                print(f"    Prompt Tokens: {usage_info.prompt_tokens}")
                print(f"    Completion Tokens: {usage_info.completion_tokens}")
                print(f"    Total Tokens: {usage_info.total_tokens}")
            else:
                print("  Token usage data not found in response_data.")
        else:
            print("  No additional response data found on api_processor.processor for the rephrasing call.")

    except Exception as e:
        print(f"\nAn error occurred during the question rephrasing process: {e}")
        print("This could be due to API key issues, network problems, LLM model errors,")
        print("or issues with the input question format.")
        import traceback
        traceback.print_exc()

    print("\nComparative question rephrasing demo complete.")

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

from src.api_requests import APIProcessor # 核心：API 请求处理器

load_dotenv() # 加载环境变量
```
- `APIProcessor`: 我们在 `demo_14` 和 `demo_15` 中已经熟悉了这个类，它封装了与 LLM API（如此处的 OpenAI API）的交互。这里我们将使用它的一个新方法 `get_rephrased_questions`。

### 2. `main()` 函数

#### 2.1. 检查 API 密钥和初始化 `APIProcessor`
这部分与 `demo_14` 和 `demo_15` 中的逻辑相同，确保 API 密钥可用并成功初始化 `APIProcessor(provider="openai")`。

#### 2.2. 定义示例比较型问题和涉及的公司
```python
    sample_comparative_question = (
        "Compare the Research and Development (R&D) expenses of 'AlphaCorp' and 'BetaInc' for the fiscal year 2023. "
        "Which company invested more in R&D during this period?"
    )
    involved_companies = ["AlphaCorp", "BetaInc"]

    print("\n--- Original Comparative Question ---")
    print(f"  Question: \"{sample_comparative_question}\"")
    print(f"  Involved Companies: {involved_companies}")
```
- `sample_comparative_question`: 提供了一个清晰的例子，该问题要求对比两家公司（'AlphaCorp' 和 'BetaInc'）在特定方面（研发费用）的表现，并进行比较判断。
- `involved_companies`: 一个列表，明确指出了问题中涉及的需要进行比较的实体（公司名称）。

#### 2.3. 改写比较型问题
```python
    print("\nRequesting rephrased questions from LLM via APIProcessor...")
    print("(This involves an LLM call using a prompt similar to RephrasedQuestionsPrompt)...")
    try:
        rephrased_map = api_processor.get_rephrased_questions(
            original_question=sample_comparative_question,
            companies=involved_companies
        )
```
- `rephrased_map = api_processor.get_rephrased_questions(...)`: **这是执行问题改写的核心调用**。
    - `original_question`: 传入我们定义的原始比较型问题。
    - `companies`: 传入涉及的公司列表。
- **内部机制**:
    - `get_rephrased_questions` 方法内部会构造一个特定的 Prompt。这个 Prompt 的设计目标是让 LLM 理解它需要将 `original_question` 分解成针对 `companies` 列表中每个公司的、独立的子问题。
    - 例如，对于 `AlphaCorp`，LLM 可能会被引导生成类似 "What were the Research and Development (R&D) expenses of 'AlphaCorp' for the fiscal year 2023?" 这样的问题。
    - 对于 `BetaInc`，则生成针对 `BetaInc` 的类似问题。
    - 这个 Prompt 很可能也是结构化的，并期望 LLM 返回一个结构化的输出（例如一个 JSON 对象，其中包含了每个公司及其对应的改写后问题）。`APIProcessor` 随后会解析这个 LLM 的输出。
- `rephrased_map`: 该方法预期返回一个 Python 字典。这个字典的键是公司名称，值是 LLM 为该公司生成的、改写后的具体子问题（字符串）。

#### 2.4. 显示改写后的问题
```python
        print("\n--- Rephrased Questions Map ---")
        if rephrased_map and isinstance(rephrased_map, dict):
            for company, rephrased_q in rephrased_map.items():
                print(f"  For Company '{company}':")
                print(f"    Rephrased Question: \"{rephrased_q}\"")
        else:
            # ... (处理未成功获取改写结果的情况) ...
```
- 脚本遍历返回的 `rephrased_map`，并为每个公司打印出其对应的、由 LLM 生成的、更具针对性的子问题。
- **预期输出示例** (基于上述 `sample_comparative_question`):
    ```
      For Company 'AlphaCorp':
        Rephrased Question: "What were the Research and Development (R&D) expenses of 'AlphaCorp' for the fiscal year 2023?"
      For Company 'BetaInc':
        Rephrased Question: "What were the Research and Development (R&D) expenses of 'BetaInc' for the fiscal year 2023?"
    ```
    (注意：LLM 可能还会生成关于“哪家公司投入更多”的原始比较部分的子问题，或者这部分会由后续的答案合成步骤处理。具体取决于 `RephrasedQuestionsPrompt` 的设计。)

#### 2.5. 检查 API 响应元数据
与 `demo_14` 和 `demo_15` 类似，这部分代码用于获取和显示执行问题改写的那次 LLM API 调用的详细信息，特别是 Token 使用量。

## 改写后的问题如何在 RAG 中使用？

得到这些针对每个实体的独立子问题后，一个完整的 RAG 系统（例如 `demo_18` 中的 `QuestionsProcessor` 在处理比较型问题时可能会采用的策略）通常会按以下步骤操作：

1.  **独立检索**: 对每一个改写后的子问题，分别执行信息检索流程（例如，调用 `HybridRetriever`），为每个公司和其相关的子问题找到最相关的文本块。
    -   例如，使用 "AlphaCorp 的研发费用是多少？" 来检索 AlphaCorp 的相关文档。
    -   使用 "BetaInc 的研发费用是多少？" 来检索 BetaInc 的相关文档。
2.  **独立答案生成/信息提取**: 对每个子问题及其检索到的上下文，分别调用 LLM（可能使用 `AnswerWithRAGContextNumberPrompt` 或类似的 Prompt）来提取或生成针对该子问题的答案。
    -   得到 AlphaCorp 的研发费用。
    -   得到 BetaInc 的研发费用。
3.  **最终答案合成**: 将所有子问题的答案（即每个公司的具体信息）收集起来，然后进行最后一次 LLM 调用（可能使用 `ComparativeAnswerPrompt`）。这个最终的调用会接收所有收集到的事实，并被要求基于这些事实来回答原始的、完整的比较型问题（包括“哪家公司投入更多？”这样的比较判断）。

通过这种“分解 -> 分别处理 -> 合成”的策略，RAG 系统能更有效地处理复杂的比较型查询。

## 如何运行脚本

1.  **设置 `OPENAI_API_KEY`**: 确保在 `.env` 文件中正确配置了你的 OpenAI API 密钥。
2.  **确保相关库已安装**: `pip install openai python-dotenv pydantic`。
3.  **打开终端或命令行工具**。
4.  **导航到脚本所在的目录**。
5.  **执行脚本**:
    ```bash
    python study/demo_19_rephrasing_comparative_questions.py
    ```
    或者，如果你在项目的根目录下：
    ```bash
    python -m study.demo_19_rephrasing_comparative_questions
    ```
    脚本将会：
    - 调用 OpenAI API。
    - 打印出 LLM 为 `AlphaCorp` 和 `BetaInc` 分别生成的、改写后的子问题。
    - 打印出该次 API 调用的 Token 使用量等元数据。

## 总结：提升 RAG 查询理解能力的关键一步

`demo_19_rephrasing_comparative_questions.py` 向我们展示了在 RAG 流程的早期阶段——**查询理解与转换**——中的一个重要高级技巧。通过利用 LLM 将复杂的用户查询（如比较型问题）分解为更简单、更易于处理的子查询，我们可以显著提高后续信息检索的准确性和相关性，并为最终生成高质量、有依据的答案打下坚实基础。

这种对用户原始查询进行预处理和优化的能力，是构建真正智能和强大的 RAG 系统的关键组成部分。希望本教程能帮助你理解并应用这一重要技术！这是我们整个系列教程的最后一篇演示，感谢您的跟随学习！
