# RAG 系统的对话艺术：`demo_13_understanding_rag_prompts.py` 之 Prompt 解析

大家好！在我们的检索增强生成（RAG）系列教程中，我们已经走过了从文档解析、索引构建到信息检索的完整流程。我们知道，RAG 的核心在于“检索”与“生成”：首先找到与用户问题最相关的信息片段（上下文），然后将这些信息连同问题一起交给大型语言模型（LLM）来“生成”答案。

那么，这个“交给 LLM”的过程是如何精确控制的呢？答案就是——**精心设计的提示（Prompts）**。本篇教程将通过 `study/demo_13_understanding_rag_prompts.py` 脚本，带大家深入理解 RAG 系统中使用的各种 Prompt 的结构和用途，特别是它们如何融合检索到的上下文，并指导 LLM 生成符合我们预期的、结构化的输出。

## 脚本目标

- 展示并解释 `src/prompts.py` 文件中定义的各种核心 Prompt 模板的结构和设计理念。
- 理解不同 Prompt 在 RAG 系统不同阶段（如问题改写、答案生成、结果重排）的作用。
- 强调在 Prompt 中如何整合用户问题、检索到的上下文信息。
- 介绍如何使用 Pydantic 模型来定义和确保 LLM 输出的 JSON 结构，从而提高系统的稳定性和可控性。

## 什么是 RAG Prompts？

RAG Prompts 并不仅仅是简单地把用户问题直接丢给 LLM。它们是经过精心构造的指令集，通常包含以下几个关键部分：

1.  **用户原始查询 (User Query)**: 用户提出的原始问题。
2.  **检索到的上下文 (Retrieved Context)**: 从知识库（例如我们之前构建的 FAISS 索引或 BM25 索引）中检索到的、与用户查询最相关的文本块。
3.  **明确的指令 (Instructions)**: 告诉 LLM 它应该扮演什么角色，如何利用提供的上下文来回答问题，以及答案应该侧重于哪些方面（例如，简洁性、特定信息点的提取、比较分析等）。
4.  **输出格式定义 (Output Format Definition)**: 为了让 LLM 的输出更易于程序化处理和保证一致性，通常会要求 LLM 以特定的格式（如 JSON）返回结果。这常常通过 Pydantic 模型来定义和强制执行。

一个设计良好的 Prompt 能够极大地提升 LLM 理解任务意图的准确性，并生成更高质量、更符合预期的答案。

## Pydantic 模型在 Prompts 中的作用

Pydantic 是一个非常流行的 Python 库，它利用 Python 的类型注解（type hints）来进行数据校验、设置管理和（在本场景中）数据结构的清晰定义。

在 RAG Prompts 的上下文中，Pydantic 模型主要用于：
-   **定义 LLM 输出的期望 JSON 结构**: 我们可以创建一个 Pydantic 模型来精确描述希望 LLM 返回的 JSON 对象应该包含哪些字段，以及每个字段的数据类型是什么。
-   **指令传递**: Prompt 中通常会包含这个 Pydantic 模型的描述或其 JSON Schema，明确告知 LLM 它需要生成符合此结构的 JSON。
-   **结果校验**: RAG 系统在收到 LLM 的 JSON 输出后，可以使用相应的 Pydantic 模型对其进行解析和校验，确保返回的数据符合预期，从而增强系统的鲁棒性。

## Python 脚本 `study/demo_13_understanding_rag_prompts.py`

这个脚本会加载并展示 `src/prompts.py` 中定义的多个 Prompt 类及其 Pydantic 输出模型的结构。让我们看一下代码：

```python
# study/demo_13_understanding_rag_prompts.py

import sys
import os
import inspect # To get source code of Pydantic models
import json # For pretty printing Pydantic schema

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prompts import (
    RephrasedQuestionsPrompt,
    AnswerWithRAGContextNamePrompt,
    AnswerWithRAGContextNumberPrompt,
    AnswerWithRAGContextBooleanPrompt,
    AnswerWithRAGContextNamesPrompt,
    ComparativeAnswerPrompt,
    RerankingPrompt,
    RetrievalRankingSingleBlock,  # Pydantic model
    RetrievalRankingMultipleBlocks # Pydantic model
)

def get_attribute_safely(prompt_obj, attr_name):
    """Safely gets an attribute from a prompt object, returning a default string if not found."""
    return getattr(prompt_obj, attr_name, "N/A (attribute not found)")

def main():
    """
    Displays and explains key prompt structures from src/prompts.py.
    These prompts are fundamental to how the RAG system interacts with LLMs
    for various tasks like question rephrasing, answer generation, and reranking.
    """
    print("--- Understanding RAG Prompts from src/prompts.py ---")
    print("This script displays the structure and purpose of various prompts used in the RAG pipeline.\n")

    # --- 1. RephrasedQuestionsPrompt ---
    print("\n--- 1. RephrasedQuestionsPrompt ---")
    print(f"  Purpose: To generate multiple rephrased versions of an initial user query. "
          f"This helps in retrieving a broader set of potentially relevant documents.")
    print(f"  Instruction:\n{get_attribute_safely(RephrasedQuestionsPrompt, 'instruction')}")
    print(f"\n  User Prompt (template):\n{get_attribute_safely(RephrasedQuestionsPrompt, 'user_prompt')}")
    print(f"\n  Pydantic Schema (source code for expected output structure):\n"
          f"{get_attribute_safely(RephrasedQuestionsPrompt, 'pydantic_schema')}")
    print(f"\n  Example (how the prompt might be filled and its expected output):\n"
          f"{get_attribute_safely(RephrasedQuestionsPrompt, 'example')}")
    print("-" * 50)

    # --- 2. AnswerWithRAGContextNamePrompt ---
    print("\n--- 2. AnswerWithRAGContextNamePrompt ---")
    print(f"  Purpose: To generate a concise answer (typically a name or short phrase) "
          f"based on a specific question and provided RAG context.")
    print(f"  Instruction:\n{get_attribute_safely(AnswerWithRAGContextNamePrompt, 'instruction')}")
    print(f"\n  User Prompt (template):\n{get_attribute_safely(AnswerWithRAGContextNamePrompt, 'user_prompt')}")
    print(f"\n  Pydantic Schema (source code):\n"
          f"{get_attribute_safely(AnswerWithRAGContextNamePrompt, 'pydantic_schema')}")
    print(f"\n  Example:\n{get_attribute_safely(AnswerWithRAGContextNamePrompt, 'example')}")
    print("-" * 50)

    # --- 3. AnswerWithRAGContextNumberPrompt ---
    print("\n--- 3. AnswerWithRAGContextNumberPrompt ---")
    print(f"  Purpose: Similar to NamePrompt, but specifically for extracting numerical answers.")
    print(f"  Instruction:\n{get_attribute_safely(AnswerWithRAGContextNumberPrompt, 'instruction')}")
    print(f"\n  User Prompt (template):\n{get_attribute_safely(AnswerWithRAGContextNumberPrompt, 'user_prompt')}")
    print(f"\n  Pydantic Schema (source code):\n"
          f"{get_attribute_safely(AnswerWithRAGContextNumberPrompt, 'pydantic_schema')}")
    print(f"\n  Example:\n{get_attribute_safely(AnswerWithRAGContextNumberPrompt, 'example')}")
    print("-" * 50)

    # --- 4. AnswerWithRAGContextBooleanPrompt ---
    print("\n--- 4. AnswerWithRAGContextBooleanPrompt ---")
    print(f"  Purpose: For questions requiring a boolean (Yes/No) answer, along with supporting evidence from the context.")
    print(f"  Instruction:\n{get_attribute_safely(AnswerWithRAGContextBooleanPrompt, 'instruction')}")
    print(f"\n  User Prompt (template):\n{get_attribute_safely(AnswerWithRAGContextBooleanPrompt, 'user_prompt')}")
    print(f"\n  Pydantic Schema (source code):\n"
          f"{get_attribute_safely(AnswerWithRAGContextBooleanPrompt, 'pydantic_schema')}")
    print(f"\n  Example:\n{get_attribute_safely(AnswerWithRAGContextBooleanPrompt, 'example')}")
    print("-" * 50)

    # --- 5. AnswerWithRAGContextNamesPrompt ---
    print("\n--- 5. AnswerWithRAGContextNamesPrompt ---")
    print(f"  Purpose: To extract a list of names or short phrases in response to a question, based on RAG context.")
    print(f"  Instruction:\n{get_attribute_safely(AnswerWithRAGContextNamesPrompt, 'instruction')}")
    print(f"\n  User Prompt (template):\n{get_attribute_safely(AnswerWithRAGContextNamesPrompt, 'user_prompt')}")
    print(f"\n  Pydantic Schema (source code):\n"
          f"{get_attribute_safely(AnswerWithRAGContextNamesPrompt, 'pydantic_schema')}")
    print(f"\n  Example:\n{get_attribute_safely(AnswerWithRAGContextNamesPrompt, 'example')}")
    print("-" * 50)

    # --- 6. ComparativeAnswerPrompt ---
    print("\n--- 6. ComparativeAnswerPrompt ---")
    print(f"  Purpose: To generate answers for comparative questions involving multiple entities or aspects, using RAG context.")
    print(f"  Instruction:\n{get_attribute_safely(ComparativeAnswerPrompt, 'instruction')}")
    print(f"\n  User Prompt (template):\n{get_attribute_safely(ComparativeAnswerPrompt, 'user_prompt')}")
    print(f"\n  Pydantic Schema (source code):\n"
          f"{get_attribute_safely(ComparativeAnswerPrompt, 'pydantic_schema')}")
    print(f"\n  Example:\n{get_attribute_safely(ComparativeAnswerPrompt, 'example')}")
    print("-" * 50)

    # --- 7. RerankingPrompt ---
    print("\n--- 7. RerankingPrompt ---")
    print(f"  Purpose: To have an LLM rerank an initial set of retrieved document chunks based on their relevance to the query. "
          f"This helps to refine the search results before final answer generation.")
    # RerankingPrompt has specific system prompts instead of a single 'instruction'.
    print(f"  System Prompt (Rerank Single Block):\n"
          f"{get_attribute_safely(RerankingPrompt, 'system_prompt_rerank_single_block')}")
    print(f"\n  System Prompt (Rerank Multiple Blocks):\n"
          f"{get_attribute_safely(RerankingPrompt, 'system_prompt_rerank_multiple_blocks')}")
    # User prompt for reranking is typically constructed dynamically with the query and context.
    # The Pydantic schemas are for the expected output structure.
    print(f"\n  Pydantic Schema (Single Block - source code):\n"
          f"{get_attribute_safely(RerankingPrompt, 'pydantic_schema_single_block')}")
    print(f"\n  Pydantic Schema (Multiple Blocks - source code):\n"
          f"{get_attribute_safely(RerankingPrompt, 'pydantic_schema_multiple_blocks')}")
    print(f"\n  Example (Single Block - how the prompt might be filled and its expected output):\n"
          f"{get_attribute_safely(RerankingPrompt, 'example_single_block')}")
    print(f"\n  Example (Multiple Blocks - how the prompt might be filled and its expected output):\n"
          f"{get_attribute_safely(RerankingPrompt, 'example_multiple_blocks')}")
    print("-" * 50)

    # --- 8. RetrievalRankingSingleBlock (Pydantic Model) ---
    print("\n--- 8. RetrievalRankingSingleBlock (Pydantic Model) ---")
    print(f"  Purpose: Defines the expected JSON output structure when an LLM reranks a single retrieved text block. "
          f"It includes fields for relevance, confidence, and reasoning.")
    try:
        # Print the source code of the Pydantic model
        schema_source = inspect.getsource(RetrievalRankingSingleBlock)
        print(f"  Pydantic Model Source Code:\n{schema_source}")
        # Alternatively, print the JSON schema:
        # print(f"  Pydantic Model JSON Schema:\n{json.dumps(RetrievalRankingSingleBlock.model_json_schema(), indent=2)}")
    except TypeError:
        print("  Could not retrieve source code for RetrievalRankingSingleBlock (likely not a class/module).")
    except Exception as e:
        print(f"  Error retrieving schema for RetrievalRankingSingleBlock: {e}")
    print("-" * 50)

    # --- 9. RetrievalRankingMultipleBlocks (Pydantic Model) ---
    print("\n--- 9. RetrievalRankingMultipleBlocks (Pydantic Model) ---")
    print(f"  Purpose: Defines the expected JSON output structure when an LLM reranks multiple retrieved text blocks. "
          f"It typically contains a list of objects, each conforming to a structure similar to RetrievalRankingSingleBlock.")
    try:
        # Print the source code of the Pydantic model
        schema_source = inspect.getsource(RetrievalRankingMultipleBlocks)
        print(f"  Pydantic Model Source Code:\n{schema_source}")
        # Alternatively, print the JSON schema:
        # print(f"  Pydantic Model JSON Schema:\n{json.dumps(RetrievalRankingMultipleBlocks.model_json_schema(), indent=2)}")
    except TypeError:
        print("  Could not retrieve source code for RetrievalRankingMultipleBlocks (likely not a class/module).")
    except Exception as e:
        print(f"  Error retrieving schema for RetrievalRankingMultipleBlocks: {e}")
    print("-" * 50)

    print("\nPrompt exploration complete.")

if __name__ == "__main__":
    main()
```

## 脚本代码详解

### 1. 导入模块
```python
import sys
import os
import inspect # 用于获取 Pydantic模型的源代码
import json    # 用于美化打印 Pydantic 的 JSON Schema (脚本中注释掉了，但可备选)

sys.path.append(...) # 添加 src 目录到 Python 路径

# 从 src.prompts 模块导入各种 Prompt 类和 Pydantic 模型
from src.prompts import (
    RephrasedQuestionsPrompt,
    AnswerWithRAGContextNamePrompt,
    # ... 其他 Prompt 类 ...
    RerankingPrompt,
    RetrievalRankingSingleBlock,  # Pydantic 模型
    RetrievalRankingMultipleBlocks # Pydantic 模型
)
```
- `inspect`: Python 标准库，`inspect.getsource()` 可以获取对象的源代码，这里用来直接展示 Pydantic 模型的定义。
- 从 `src.prompts` 导入的类分为两大种：
    - **Prompt 类**: 如 `RephrasedQuestionsPrompt`, `AnswerWithRAGContextNamePrompt` 等。这些类通常封装了与 LLM 交互所需的各种文本片段（指令、用户输入模板、示例等）以及期望的输出结构（Pydantic schema）。
    - **Pydantic 模型类**: 如 `RetrievalRankingSingleBlock`, `RetrievalRankingMultipleBlocks`。这些是纯粹的 Pydantic 模型，直接定义了 LLM 在某些任务（如重排序）中应返回的 JSON 数据的结构。

### 2. `get_attribute_safely` 辅助函数
```python
def get_attribute_safely(prompt_obj, attr_name):
    """Safely gets an attribute from a prompt object, returning a default string if not found."""
    return getattr(prompt_obj, attr_name, "N/A (attribute not found)")
```
- 这是一个简单的小工具函数，用于安全地获取一个对象（这里是 Prompt 对象）的属性。如果属性不存在，它会返回一个默认的提示字符串，而不是抛出 `AttributeError` 导致程序中断。

### 3. `main()` 函数：遍历并展示各种 Prompt
`main()` 函数逐一实例化或引用 `src.prompts` 中定义的各种 Prompt 类和 Pydantic 模型，并打印出它们的关键组成部分。

对于每个 **Prompt 类** (例如 `RephrasedQuestionsPrompt`, `AnswerWithRAGContextNamePrompt` 等)，脚本通常会展示：

-   **`Purpose` (用途)**:
    -   脚本直接用 print 语句解释了这个 Prompt 在 RAG 系统中的主要作用。例如，`RephrasedQuestionsPrompt` 用于生成用户原始问题的多种改写版本，以期通过这些改写版本从知识库中检索到更全面、更相关的信息。
-   **`Instruction` (指令)**:
    -   `get_attribute_safely(PromptClass, 'instruction')`
    -   这通常是 Prompt 中最重要的部分之一，它作为“系统消息”或高级指令，告诉 LLM 它应该如何行动，遵循什么规则，以什么角色自居等。例如，对于答案生成类的 Prompt，指令可能会要求 LLM“你是一个专业的助手，请根据提供的上下文回答问题，如果上下文中没有相关信息，请明确指出”。
-   **`User Prompt (template)` (用户提示模板)**:
    -   `get_attribute_safely(PromptClass, 'user_prompt')`
    -   这是构成完整 Prompt 的用户输入部分。它通常是一个**模板字符串**，包含了占位符，如 `{question}` 和 `{context}`。在实际调用 LLM 之前，这些占位符会被替换成用户的具体问题和从 RAG 系统检索到的上下文信息。
    -   例如：`"Question: {question}\nContext: {context}\nAnswer:"`
-   **`Pydantic Schema (source code for expected output structure)` (Pydantic Schema)**:
    -   `get_attribute_safely(PromptClass, 'pydantic_schema')`
    -   对于那些期望 LLM 返回结构化 JSON 输出的 Prompt，这个属性（如果存在）通常会包含定义该 JSON 结构的 Pydantic 模型的**源代码字符串**或者该 Pydantic 模型的类本身。这明确地告诉了 LLM（或者更准确地说，是使用这个 Prompt 的开发者）所期望的输出格式。LLM 会被指示生成符合此 schema 的 JSON。
-   **`Example` (示例)**:
    -   `get_attribute_safely(PromptClass, 'example')`
    -   这部分展示了一个该 Prompt 被填充（即占位符被具体内容替换）后的完整示例，以及（通常）一个符合其 `pydantic_schema` 的预期输出示例。这对于理解 Prompt 如何工作以及 LLM 应该如何回应非常有帮助，也常用于 Few-Shot Learning 场景。

**特殊情况：`RerankingPrompt`**
-   `RerankingPrompt` 用于让 LLM 对初步检索到的文本块列表进行“重排序”，以进一步提升最相关块的排序位置。
-   它可能包含多个不同的系统提示 (`system_prompt_rerank_single_block`, `system_prompt_rerank_multiple_blocks`)，分别对应不同的重排策略（例如，是让 LLM 逐个评估块的相关性，还是同时比较多个块）。
-   它也关联了不同的 Pydantic Schema (`pydantic_schema_single_block`, `pydantic_schema_multiple_blocks`) 来定义不同重排策略下的期望输出。

对于直接展示的 **Pydantic 模型类** (例如 `RetrievalRankingSingleBlock`, `RetrievalRankingMultipleBlocks`)，脚本会：
-   **`Purpose` (用途)**: 解释这个 Pydantic 模型定义了哪种类型的结构化数据。例如，`RetrievalRankingSingleBlock` 可能定义了对单个文本块进行重排序评估后应包含的字段，如 `is_relevant` (布尔值), `confidence_score` (浮点数), `reasoning` (字符串解释)等。
-   **`Pydantic Model Source Code` (Pydantic 模型源代码)**:
    -   `inspect.getsource(PydanticModelClass)`
    -   脚本使用 `inspect.getsource()` 直接获取并打印出该 Pydantic 模型类的 Python 源代码。这使得我们可以非常清晰地看到模型包含哪些字段、每个字段的类型注解是什么、是否有默认值、是否有校验器等。这是理解期望输出结构最直接的方式。
    -   脚本中也注释了另一种显示方式：`json.dumps(PydanticModelClass.model_json_schema(), indent=2)`，这会打印出该 Pydantic 模型对应的 JSON Schema 表示，这也是一种标准的结构定义方式。

### 示例解读：`AnswerWithRAGContextNamePrompt`

假设 `AnswerWithRAGContextNamePrompt` 的结构如下（在 `src/prompts.py` 中定义）：
```python
# (src/prompts.py 中的一个虚构示例)
# class OutputName(BaseModel):
#     name: str = Field(description="The extracted name.")

# class AnswerWithRAGContextNamePrompt:
#     instruction = "You are an expert at extracting specific information. Please extract the required name from the context."
#     user_prompt = "Context: {context}\nQuestion: {question}\nBased on the context, what is the name of the entity discussed?"
#     pydantic_schema = OutputName # 或者 inspect.getsource(OutputName)
#     example = """
#     Filled Prompt:
#     Context: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
#     Question: What is the name of the tower mentioned?
#     Based on the context, what is the name of the entity discussed?

#     Expected JSON Output:
#     {"name": "Eiffel Tower"}
#     """
```
当 `demo_13_understanding_rag_prompts.py` 运行到处理 `AnswerWithRAGContextNamePrompt` 时，它会打印出：
- **Purpose**: 解释其用于提取名称。
- **Instruction**: "You are an expert..."
- **User Prompt (template)**: "Context: {context}\nQuestion: {question}..."
- **Pydantic Schema**: `OutputName` 类的源代码（或其引用）。
- **Example**: 上述包含已填充提示和预期 JSON 输出的完整示例。

## 关键启示

1.  **Prompt 的结构化和模块化**: RAG 系统中的 Prompt 不是随意编写的，而是根据特定任务（问题改写、答案生成、信息提取、重排序等）精心设计和模块化的。
2.  **指令的重要性**: `instruction` 部分对 LLM 的行为起着关键的引导作用。
3.  **上下文的动态填充**: `user_prompt` 中的 `{context}` 和 `{question}` 占位符是 RAG 的核心，它们在运行时被动态替换。
4.  **Pydantic 强制输出结构**: 使用 Pydantic schema 来定义和期望 LLM 的输出为特定结构的 JSON，极大地增强了 RAG 系统的稳定性和后续数据处理的便捷性。LLM 被训练（或通过指令引导）来遵循这种结构。
5.  **示例的引导作用**: `example` 不仅帮助开发者理解 Prompt，也可能在 few-shot prompting 场景中直接提供给 LLM 作为参考。

## 如何运行脚本

1.  确保你的 Python 环境中已安装必要的库（如 `pydantic`, `openai`，尽管此脚本本身不执行 OpenAI 调用，但 `src.prompts` 中的类可能导入它们）。
2.  **打开终端或命令行工具**。
3.  **导航到脚本所在的目录** (即 `study/` 目录，或者包含 `study` 目录的项目根目录)。
4.  **执行脚本**:
    ```bash
    python study/demo_13_understanding_rag_prompts.py
    ```
    或者，如果你在项目的根目录下：
    ```bash
    python -m study.demo_13_understanding_rag_prompts
    ```
    脚本将会在控制台打印出 `src/prompts.py` 中定义的各种 Prompt 结构及其 Pydantic 输出模型的详细信息。这个脚本的主要目的是展示和解释，而不是执行一个完整的 RAG 流程。

## 总结

`demo_13_understanding_rag_prompts.py` 为我们揭示了 RAG 系统与 LLM “对话”的艺术。通过理解这些精心设计的 Prompt 模板——它们的指令、用户输入结构、上下文融合方式以及通过 Pydantic 实现的结构化输出要求——我们能更好地把握如何有效地引导 LLM 在 RAG 流程中完成特定任务。

掌握 Prompt 工程（Prompt Engineering）和结构化输出处理，是构建高效、可靠、可控的 RAG 应用的关键技能。希望这篇教程能帮助你深入理解 RAG 系统中 Prompt 的核心作用！
