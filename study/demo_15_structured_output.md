# RAG 进阶：`demo_15_structured_output.py` 之获取结构化 JSON 输出

大家好！在 `demo_14` 中，我们学习了如何通过 `APIProcessor` 向 OpenAI LLM 发送请求并获取文本回复。这对于简单的问答是可行的，但在更复杂的应用中，我们往往希望 LLM 返回的不仅仅是纯文本，而是具有特定结构的 JSON 数据。这样，我们的程序就能更轻松、更可靠地解析和使用这些信息。

本篇教程将通过 `study/demo_15_structured_output.py` 脚本，向大家展示如何利用 Pydantic 模型来定义我们期望的 JSON 输出结构，并指导 LLM（如 GPT-4o-mini）按照这个结构返回 JSON 对象。`APIProcessor` 会帮助我们处理这个过程，并将 LLM 返回的 JSON 字符串自动解析为 Python 字典。

## 脚本目标

- 演示如何定义一个 Pydantic 模型来描述期望的 JSON 输出结构。
- 展示如何使用 `APIProcessor` 的特定参数（`is_structured=True`, `response_format=<PydanticModel>`）来请求结构化的 JSON 输出。
- 理解 LLM 是如何被引导（通常通过其 API 的 JSON 模式或函数调用特性）来生成符合 Pydantic 模型 schema 的 JSON 的。
- 如何直接使用从 `APIProcessor` 返回的、已解析为 Python 字典的结构化数据。

## 什么是 Pydantic 与结构化输出？

**Pydantic** 是一个 Python 库，它使用 Python 的类型注解来进行数据验证和设置管理。在与 LLM 交互的场景中，我们可以用 Pydantic 来：

1.  **定义数据模式 (Schema)**: 创建一个 Pydantic 模型（一个继承自 `pydantic.BaseModel` 的类），在其中定义我们希望 LLM 输出的 JSON 对象应该包含哪些字段（比如 `answer`, `confidence_score`），以及每个字段的预期数据类型（如 `str`, `float`, `list[str]`）。还可以为字段添加描述。
2.  **指导 LLM 生成 JSON**: 当我们向 LLM API（如 OpenAI API）发起请求时，可以通过特定的方式（例如，OpenAI API 的 "JSON Mode" 或以前的 "function calling" / "tool use" 功能）将 Pydantic 模型的 JSON Schema（一个描述模型结构的元数据文档）传递给 LLM。这样，LLM 就知道它需要生成一个符合此特定结构的 JSON 字符串。
3.  **自动解析和验证**: `APIProcessor` 在收到 LLM 返回的 JSON 字符串后，如果配置正确（如此 demo 所示），它可以自动将这个 JSON 字符串解析成 Python 字典。如果使用了 Pydantic 模型实例进行解析，Pydantic 还会自动验证返回的数据是否符合模型定义的类型和约束。

**这样做的好处是什么？**
-   **可靠性**: 确保 LLM 的输出是我们程序可以稳定处理的格式，减少因 LLM 输出格式多变导致的错误。
-   **易用性**: 直接得到 Python 字典或 Pydantic 模型实例，无需编写复杂的字符串解析逻辑。
-   **开发效率**: 清晰的数据结构定义使得前后端（或不同模块间）的协作更顺畅。

## 前提条件

1.  **`OPENAI_API_KEY` 环境变量**: 必须设置你的 OpenAI API 密钥。
2.  **Pydantic 基础**: 对 Python 类和 Pydantic 模型的基本定义有所了解会很有帮助。
3.  **LLM 模型支持**: 确保你使用的 LLM 模型支持 JSON 输出模式或类似功能。较新的 OpenAI 模型（如 `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo` 的一些版本）通常有良好的支持。

## Python 脚本 `study/demo_15_structured_output.py`

让我们完整地看一下这个脚本的代码：
```python
# study/demo_15_structured_output.py

import sys
import os
import json
from pathlib import Path # Not strictly used but good practice for path handling
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api_requests import APIProcessor

# Load environment variables from .env file (especially OPENAI_API_KEY)
load_dotenv()

# --- Purpose of this Demo ---
# This script demonstrates how to use the `APIProcessor` class to request
# and receive structured JSON output from an OpenAI Language Model (LLM).
# By providing a Pydantic model as the desired `response_format`, the LLM
# is instructed (often via function calling or JSON mode) to generate output
# that conforms to the schema of that Pydantic model. `APIProcessor` then
# automatically parses this JSON output into a Python dictionary.
#
# IMPORTANT:
# An `OPENAI_API_KEY` must be set in your .env file in the project root.
# Structured output capabilities may depend on the specific LLM model used
# (e.g., "gpt-4o-mini", "gpt-4-turbo" support JSON mode well).

# --- 1. Define Pydantic Model for Structured Output ---
class SimpleResponse(BaseModel):
    answer: str = Field(description="The direct answer to the question.")
    confidence_score: float = Field(description="A score from 0.0 to 1.0 indicating confidence.")
    related_topics: list[str] = Field(description="A list of related topics.")

def main():
    """
    Demonstrates sending a request to an OpenAI LLM and receiving
    a structured JSON response parsed according to a Pydantic model.
    """
    print("Starting structured output (JSON with Pydantic) demo...")

    # --- 2. Check for API Key ---
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please ensure your API key is configured in the .env file in the project root.")
        return
    print("OPENAI_API_KEY found in environment.")

    # --- 3. Initialize APIProcessor ---
    try:
        api_processor = APIProcessor(provider="openai")
        print("APIProcessor initialized for OpenAI.")
    except Exception as e:
        print(f"Error initializing APIProcessor: {e}")
        return

    # --- 4. Define Request Components ---
    question = "What is the capital of France and what are two related topics regarding its history?"
    
    # System content guides the LLM. When using Pydantic models for response_format with OpenAI,
    # the APIProcessor (or underlying OpenAI client) typically appends the JSON schema of the
    # Pydantic model to the system message or uses tools/function-calling to enforce the structure.
    # This system_content is a general instruction.
    system_content = (
        "You are an AI assistant. Answer the user's question. You must provide a direct answer, "
        "a confidence score (from 0.0 to 1.0), and a list of two related topics. "
        "Format your response according to the provided schema."
    )

    print("\n--- Request Details ---")
    print(f"  Question: \"{question}\"")
    print(f"  System Content Hint (schema is also sent by APIProcessor):\n    \"{system_content}\"")
    print(f"  Expected Pydantic Schema: {SimpleResponse.__name__}")

    # --- 5. Send Request to LLM for Structured Output ---
    llm_model_name = "gpt-4o-mini" # Or "gpt-4-turbo", "gpt-3.5-turbo" (check model's JSON mode support)
    print(f"\nSending request to OpenAI model: {llm_model_name} for structured output...")

    try:
        # `is_structured=True` and `response_format=SimpleResponse` (the Pydantic model class)
        # signals APIProcessor to configure the request for structured JSON output.
        response_dict = api_processor.send_message(
            model=llm_model_name,
            system_content=system_content,
            human_content=question,
            temperature=0.1,
            is_structured=True,
            response_format=SimpleResponse # Pass the Pydantic model class
        )

        print("\n--- LLM Response ---")
        print(f"  Original Question: {question}")

        if response_dict and isinstance(response_dict, dict):
            print("\n  Structured LLM Response (parsed as dictionary by APIProcessor):")
            print(json.dumps(response_dict, indent=2))

            print("\n  Accessing individual fields from the dictionary:")
            print(f"    Answer: {response_dict.get('answer')}")
            print(f"    Confidence Score: {response_dict.get('confidence_score')}")
            related = response_dict.get('related_topics', [])
            print(f"    Related Topics: {', '.join(related) if related else 'N/A'}")
        elif response_dict: # If not a dict but something was returned
            print("\n  Received a non-dictionary response (unexpected for structured output):")
            print(f"    Type: {type(response_dict)}")
            print(f"    Content: {response_dict}")
        else:
            print("\n  Failed to get a structured response or response was None.")

        # --- 6. Inspect Raw API Response Data ---
        print("\n--- API Response Metadata (from api_processor.processor.response_data) ---")
        if hasattr(api_processor.processor, 'response_data') and api_processor.processor.response_data:
            response_metadata = api_processor.processor.response_data
            if hasattr(response_metadata, 'model'):
                 print(f"  Model Used (from API): {response_metadata.model}")
            if hasattr(response_metadata, 'usage') and response_metadata.usage:
                usage_info = response_metadata.usage
                print(f"  Token Usage: Prompt={usage_info.prompt_tokens}, Completion={usage_info.completion_tokens}, Total={usage_info.total_tokens}")
            else:
                print("  Token usage data not found in response_data.")
        else:
            print("  No additional response data found on api_processor.processor.")

    except Exception as e:
        print(f"\nAn error occurred during the API request for structured output: {e}")
        print("This could be due to issues with the LLM's ability to conform to the schema,")
        print("API key problems, network issues, or model limitations.")
        import traceback
        traceback.print_exc()

    print("\nStructured output demo complete.")

if __name__ == "__main__":
    main()
```

## 脚本代码详解

### 1. 导入模块
```python
import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field # Pydantic 核心组件

sys.path.append(...) # 添加 src 目录

from src.api_requests import APIProcessor # API 请求处理器

load_dotenv() # 加载 .env 文件
```
- `pydantic.BaseModel` 和 `pydantic.Field`: 这是定义 Pydantic 模型的基础。`BaseModel` 是所有 Pydantic模型的父类，`Field` 用于为模型字段添加额外的元数据（如描述、默认值、校验规则等）。

### 2. 定义 Pydantic 模型 (`SimpleResponse`)
```python
class SimpleResponse(BaseModel):
    answer: str = Field(description="The direct answer to the question.")
    confidence_score: float = Field(description="A score from 0.0 to 1.0 indicating confidence.")
    related_topics: list[str] = Field(description="A list of related topics.")
```
- 这段代码定义了一个名为 `SimpleResponse` 的 Pydantic 模型，它描述了我们期望 LLM 返回的 JSON 对象应该具有的结构：
    - `answer`: 一个字符串类型（`str`）的字段，用于存放问题的直接答案。`Field(description=...)` 为这个字段添加了描述，这个描述信息可能会被 `APIProcessor` 用来生成传递给 LLM 的 JSON Schema 的一部分，帮助 LLM 理解每个字段的含义。
    - `confidence_score`: 一个浮点数类型（`float`）的字段，表示 LLM 对其答案的置信度，范围从 0.0 到 1.0。
    - `related_topics`: 一个字符串列表类型（`list[str]`）的字段，用于存放与问题相关的几个主题。

### 3. `main()` 函数

#### 3.1. API 密钥检查与 `APIProcessor` 初始化
这部分与 `demo_14` 中的逻辑相同，确保 API 密钥可用并初始化 `APIProcessor`。

#### 3.2. 定义请求组件
```python
    question = "What is the capital of France and what are two related topics regarding its history?"
    system_content = (
        "You are an AI assistant. Answer the user's question. You must provide a direct answer, "
        "a confidence score (from 0.0 to 1.0), and a list of two related topics. "
        "Format your response according to the provided schema." # <--- 重要提示
    )
    print(f"  Expected Pydantic Schema: {SimpleResponse.__name__}")
```
- `question`: 用户的具体问题。
- `system_content`: 系统指令。与 `demo_14` 相比，这里的指令**额外强调了“Format your response according to the provided schema.”**（请根据提供的 schema 格式化你的响应）。
- **重要说明**: 虽然这里的 `system_content` 做了提示，但 `APIProcessor`（或其底层的 OpenAI Python 客户端库）在收到 `response_format=SimpleResponse` 参数时，通常会自动获取 `SimpleResponse` 模型的 JSON Schema，并将其以 LLM 能理解的方式（例如，通过特定的系统消息、OpenAI API 的 `tools` 或 `tool_choice` 参数，或启用 "JSON mode"）传递给 LLM。这样 LLM 才能确切知道要生成的 JSON 的具体字段、类型和结构。

#### 3.3. 发送请求以获取结构化输出
```python
    llm_model_name = "gpt-4o-mini"
    print(f"\nSending request to OpenAI model: {llm_model_name} for structured output...")
    try:
        response_dict = api_processor.send_message(
            model=llm_model_name,
            system_content=system_content,
            human_content=question, # 注意：这里没有显式地将 context 传入，因为问题本身不依赖外部RAG上下文
            temperature=0.1,
            is_structured=True,          # <--- 关键参数
            response_format=SimpleResponse # <--- 关键参数：传递 Pydantic 模型类
        )
```
- `llm_model_name`: 选择支持结构化输出（如 JSON Mode）的 LLM 模型。
- `api_processor.send_message(...)`:
    - `is_structured=True`: 这个布尔标志告诉 `APIProcessor` 我们期望一个结构化的响应，而不是纯文本。
    - `response_format=SimpleResponse`: **这是实现结构化输出的核心**。我们将之前定义的 `SimpleResponse` Pydantic **模型类本身**传递给这个参数。`APIProcessor` 会利用这个模型信息来配置对 OpenAI API 的调用，以便指示 LLM 返回符合 `SimpleResponse` schema 的 JSON。
- `response_dict`: 如果一切顺利，`APIProcessor.send_message` 方法在收到 LLM 返回的 JSON 字符串后，会**自动将其解析为一个 Python 字典**，并赋值给 `response_dict`。

#### 3.4. 使用结构化的响应
```python
        if response_dict and isinstance(response_dict, dict):
            print("\n  Structured LLM Response (parsed as dictionary by APIProcessor):")
            print(json.dumps(response_dict, indent=2)) # 美化打印整个字典

            print("\n  Accessing individual fields from the dictionary:")
            print(f"    Answer: {response_dict.get('answer')}")
            print(f"    Confidence Score: {response_dict.get('confidence_score')}")
            related = response_dict.get('related_topics', [])
            print(f"    Related Topics: {', '.join(related) if related else 'N/A'}")
        # ... (其他情况处理) ...
```
- **直接作为字典使用**: 由于 `APIProcessor` 已经将 LLM 返回的 JSON 字符串解析成了 Python 字典 `response_dict`，我们可以直接通过键来访问其中的数据，例如 `response_dict.get('answer')`。
- **对比**: 相比于从纯文本回复中用正则表达式或字符串查找来提取信息，这种方式无疑更简单、更健壮。
- **Pydantic 实例 (可选的下一步)**: 虽然 `APIProcessor` 在这个例子中返回的是字典，但在更完整的实现中，它甚至可以直接返回 `SimpleResponse` 的一个实例 (`SimpleResponse(**response_dict)`)，这样我们就可以通过属性（如 `response_obj.answer`）来访问数据，并且所有数据都经过了 Pydantic 的类型校验。

#### 3.5. 检查 API 响应元数据
这部分与 `demo_14` 类似，用于获取和显示如 Token 使用量等 API 调用相关的元数据。

## 关键启示

1.  **Pydantic 定义契约**: Pydantic 模型为我们和 LLM之间定义了一个清晰的数据交换“契约”。我们明确了期望的数据结构，LLM 则被引导去遵循这个契约。
2.  **简化 LLM 交互**: `APIProcessor` 结合 Pydantic 模型，使得请求和处理结构化 JSON 输出的过程大大简化。开发者无需手动处理 JSON Schema 的生成或复杂的响应解析。
3.  **提高鲁棒性**: 当 LLM 的输出是结构化的并且经过验证时，应用程序的后续处理逻辑会更加稳定，不易因 LLM 输出的微小文本变化而出错。
4.  **适用于复杂数据提取**: 对于需要从文本中提取多个相关信息点并以特定格式组织的任务，这种方法非常强大。

## 如何运行脚本

1.  **设置 `OPENAI_API_KEY`**: 确保在 `.env` 文件中正确配置了你的 OpenAI API 密钥。
2.  **确保相关库已安装**: `pip install openai python-dotenv pydantic`。
3.  **打开终端或命令行工具**。
4.  **导航到脚本所在的目录**。
5.  **执行脚本**:
    ```bash
    python study/demo_15_structured_output.py
    ```
    或者，如果你在项目的根目录下：
    ```bash
    python -m study.demo_15_structured_output
    ```
    脚本将会：
    - 调用 OpenAI API（例如 `gpt-4o-mini` 模型）。
    - 指示 LLM 根据 `SimpleResponse` Pydantic 模型的 schema 返回 JSON。
    - 打印出 LLM 返回的、已解析为 Python 字典的结构化数据。
    - 打印出 API 响应中的元数据（如 Token 使用量）。

## 总结

`demo_15_structured_output.py` 演示了现代 LLM 应用开发中的一个重要实践：如何从 LLM 获取可靠的、程序易于处理的结构化 JSON 输出。通过结合 Pydantic 模型来定义数据模式，并利用像 `APIProcessor` 这样的工具类来封装与 LLM API 的交互（特别是其支持 JSON 输出的特性），我们可以构建出更强大、更稳定的 AI 应用。

这不仅简化了开发流程，也为处理复杂的信息提取和生成任务提供了坚实的基础。希望本教程能帮助你掌握这一关键技能！
