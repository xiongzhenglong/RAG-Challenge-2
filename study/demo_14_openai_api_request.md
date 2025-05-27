# RAG 实战生成篇：`demo_14_openai_api_request.py` 之调用 OpenAI API 获取答案

大家好！在我们的检索增强生成（RAG）系列教程中，我们已经深入学习了如何从 PDF 文档中提取信息、构建检索索引（FAISS 和 BM25），并实际执行了检索操作以找到与用户问题相关的文本块。现在，我们终于来到了 RAG 流程中激动人心的“G”——**生成 (Generation)** 环节。

本篇教程将通过 `study/demo_14_openai_api_request.py` 脚本，向大家展示如何使用 `src.api_requests.APIProcessor` 类，将用户的问题和我们检索到的上下文信息一起发送给 OpenAI 的大型语言模型（LLM，例如 GPT-4o-mini），并获取模型基于这些信息生成的答案。这将是一个基础但完整的 RAG 交互演示。

## 脚本目标

- 演示如何使用 `APIProcessor` 类与 OpenAI LLM（如 GPT-4o-mini）进行交互。
- 展示一个基本的 RAG 式请求流程：如何将用户问题与检索到的上下文（RAG Context）结合，形成发送给 LLM 的提示（Prompt）。
- 理解发送给 LLM 的不同类型内容：系统指令（System Content）和用户内容（Human Content）。
- 如何解析 LLM 返回的响应，包括生成的文本答案以及 API 返回的元数据（如 Token 使用量）。
- **强调 `OPENAI_API_KEY` 在环境变量中配置的重要性。**

## 理解对 LLM 的 API 请求

当我们说“调用 LLM API”时，通常指的是向一个远程的 LLM 服务（如 OpenAI 的服务器）发送一个结构化的请求。这个请求主要包含：

1.  **模型名称 (Model Name)**: 指定你希望使用哪个 LLM，例如 `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo` 等。不同的模型在能力、速度和价格上有所不同。
2.  **提示内容 (Prompts)**: 这是与 LLM 沟通的核心。通常分为：
    -   **系统内容/指令 (System Content / System Message)**: 给 LLM 的高级指令，设定其角色、行为准则和回答问题的约束。在 RAG 中，这部分非常重要，例如可以指示 LLM “你是一个乐于助人的助手，请**仅根据**提供的上下文来回答问题，如果答案在上下文中找不到，请明确说明。” 这样做可以有效减少模型的“幻觉”（即编造信息）。
    -   **用户内容/人类消息 (Human Content / Human Message)**: 包含用户具体的问题，以及从我们知识库中检索到的、与问题最相关的上下文信息（RAG Context）。
3.  **参数 (Parameters)**:
    -   `temperature`: 控制输出的随机性。较低的值（如 0.1-0.3）使输出更具确定性和一致性；较高的值（如 0.7-1.0）则使输出更具创造性和多样性。
    -   `max_tokens`: （可选）限制 LLM 生成答案的最大长度（以 token 为单位）。

LLM 服务在收到请求后，会处理这些信息并返回一个响应。

## 解读 LLM 的响应

LLM API 的响应通常包含：

1.  **生成的文本内容 (Generated Text)**: 这是 LLM 根据你的 Prompt 生成的直接答案或内容。
2.  **元数据 (Metadata)**:
    -   **模型信息**: 确认实际使用了哪个模型。
    -   **Token 使用量**: 这非常重要！LLM 的计费通常基于处理的 token 数量。API 响应会告诉你这次请求中：
        -   `prompt_tokens`: 输入提示（系统内容 + 用户内容）所消耗的 token 数。
        -   `completion_tokens`: LLM 生成的答案所消耗的 token 数。
        -   `total_tokens`: 总消耗 token 数。
    -   了解 token 使用量有助于管理 API 调用成本和避免超出模型的上下文窗口限制。

## 前提条件

- **`OPENAI_API_KEY` 环境变量**: **至关重要！** 你必须在项目的根目录下创建一个 `.env` 文件，并在其中设置你的 OpenAI API 密钥。关于如何设置，请参考 `study/demo_01_project_setup.py` 教程中的说明。没有这个密钥，脚本无法与 OpenAI API 通信。

## Python 脚本 `study/demo_14_openai_api_request.py`

让我们完整地看一下这个脚本的代码：
```python
# study/demo_14_openai_api_request.py

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api_requests import APIProcessor

# Load environment variables from .env file (especially OPENAI_API_KEY)
load_dotenv()

# --- Purpose of this Demo ---
# This script demonstrates how to use the `APIProcessor` class to interact
# with an OpenAI Language Model (LLM) like GPT-4o-mini.
# It shows a basic Request-Augmented Generation (RAG) pattern where a question
# is answered based on provided context.
#
# IMPORTANT:
# An `OPENAI_API_KEY` must be set in your .env file in the project root
# for this script to work. Refer to `study/demo_01_project_setup.py`
# for instructions on setting up your .env file and API keys.

def main():
    """
    Demonstrates sending a request to an OpenAI LLM using APIProcessor
    with a sample question and RAG context.
    """
    print("Starting OpenAI API request demo...")

    # --- 1. Check for API Key ---
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please ensure your API key is configured in the .env file in the project root.")
        print("Refer to 'study/demo_01_project_setup.py' for guidance.")
        return

    print("OPENAI_API_KEY found in environment.")

    # --- 2. Initialize APIProcessor ---
    try:
        # APIProcessor can be configured for different providers (e.g., "openai", "google", "ibm")
        # It handles the underlying client initialization and request formatting.
        api_processor = APIProcessor(provider="openai")
        print("APIProcessor initialized for OpenAI.")
    except Exception as e:
        print(f"Error initializing APIProcessor: {e}")
        return

    # --- 3. Define Request Components ---
    question = "What is the main product of ExampleCorp?"
    rag_context = (
        "ExampleCorp is a leading provider of advanced widget solutions. "
        "Their flagship product, the 'SuperWidget', is known for its "
        "efficiency and reliability. ExampleCorp also offers consulting services "
        "for widget integration."
    )

    # System content guides the LLM's behavior and tone.
    system_content = (
        "You are a helpful assistant. Your task is to answer the user's question "
        "based *only* on the provided context. If the answer cannot be found within "
        "the context, you must explicitly state 'Information not found in the provided context.' "
        "Do not make up information or use external knowledge."
    )

    # Human content combines the context and the specific question.
    human_content = f"Context:\n---\n{rag_context}\n---\n\nQuestion: {question}"

    print("\n--- Request Details ---")
    print(f"  System Content (Instructions to LLM):\n    \"{system_content}\"")
    print(f"\n  Human Content (Context + Question):\n    Context: \"{rag_context[:100]}...\""
          f"\n    Question: \"{question}\"")

    # --- 4. Send Request to LLM ---
    # We use "gpt-4o-mini" as it's a fast and cost-effective model suitable for many tasks.
    # Other models like "gpt-4-turbo" or "gpt-3.5-turbo" can also be used.
    # Temperature controls randomness: 0.1 makes the output more deterministic.
    llm_model_name = "gpt-4o-mini"
    print(f"\nSending request to OpenAI model: {llm_model_name}...")

    try:
        response = api_processor.send_message(
            model=llm_model_name,
            system_content=system_content,
            human_content=human_content,
            temperature=0.1,
            # max_tokens=100 # Optionally control max output length
        )

        print("\n--- LLM Response ---")
        print(f"  Question: {question}")
        print(f"  Provided RAG Context: \"{rag_context}\"")
        print(f"\n  LLM's Answer:\n    \"{response}\"")

        # --- 5. Inspect Response Data (Token Usage, Model Info) ---
        # The `api_processor.processor.response_data` attribute (if available on the
        # specific processor like `OpenAIProcessor`) often stores the raw response object
        # from the API, which includes metadata like token usage.
        print("\n--- API Response Metadata (from api_processor.processor.response_data) ---")
        if hasattr(api_processor.processor, 'response_data') and api_processor.processor.response_data:
            response_metadata = api_processor.processor.response_data
            
            # Print the model used (as reported by the API)
            if hasattr(response_metadata, 'model'):
                 print(f"  Model Used (from API): {response_metadata.model}")

            # Print token usage if available
            if hasattr(response_metadata, 'usage') and response_metadata.usage:
                usage_info = response_metadata.usage
                print(f"  Token Usage:")
                print(f"    Prompt Tokens: {usage_info.prompt_tokens}")
                print(f"    Completion Tokens: {usage_info.completion_tokens}")
                print(f"    Total Tokens: {usage_info.total_tokens}")
            else:
                print("  Token usage data not found in response_data.")
            
            # You can print the whole object too, but it can be verbose
            # print(f"  Full response_data object:\n{response_metadata}")
        else:
            print("  No additional response data found on api_processor.processor.")

    except Exception as e:
        print(f"\nAn error occurred during the API request: {e}")
        print("This could be due to various reasons such as:")
        print("  - Incorrect API key or insufficient credits.")
        print("  - Network connectivity issues.")
        print("  - Issues with the requested model (e.g., availability, rate limits).")
        print("  - Problems with the input data format or length.")
        import traceback
        traceback.print_exc()

    print("\nOpenAI API request demo complete.")

if __name__ == "__main__":
    main()
```

## 脚本代码详解

### 1. 导入模块
```python
import sys
import os
import json # 虽然本脚本未直接使用，但与LLM交互经常涉及JSON
from pathlib import Path
from dotenv import load_dotenv # 用于从.env文件加载环境变量

sys.path.append(...) # 添加 src 目录到 Python 路径

from src.api_requests import APIProcessor # 核心：API 请求处理器

load_dotenv() # 脚本开始时即加载 .env 文件
```
- `dotenv.load_dotenv()`: **重要**！此函数会查找项目根目录下的 `.env` 文件，并将其中的键值对加载为环境变量。我们主要依赖它来加载 `OPENAI_API_KEY`。
- `APIProcessor`: 这是我们自己封装的类（在 `src/api_requests.py` 中定义），它简化了与不同 LLM 提供商（如 OpenAI, Google, IBM）API 的交互。

### 2. `main()` 函数

#### 2.1. 检查 API 密钥
```python
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        # ... (错误提示并退出) ...
        return
    print("OPENAI_API_KEY found in environment.")
```
- 脚本首先检查 `OPENAI_API_KEY` 是否已通过 `.env` 文件（或其他方式）加载到环境变量中。如果未找到，则无法继续。

#### 2.2. 初始化 `APIProcessor`
```python
    try:
        api_processor = APIProcessor(provider="openai")
        print("APIProcessor initialized for OpenAI.")
    except Exception as e:
        # ... (初始化错误处理) ...
        return
```
- `api_processor = APIProcessor(provider="openai")`: 创建 `APIProcessor` 实例，并明确指定提供商为 `"openai"`。`APIProcessor` 内部会根据这个提供商信息来初始化相应的 LLM 客户端（例如，OpenAI 的 `OpenAI` 客户端）。

#### 2.3. 定义请求组件
```python
    question = "What is the main product of ExampleCorp?"
    rag_context = (
        "ExampleCorp is a leading provider of advanced widget solutions. "
        "Their flagship product, the 'SuperWidget', is known for its "
        "efficiency and reliability. ExampleCorp also offers consulting services "
        "for widget integration."
    )
    system_content = (
        "You are a helpful assistant. Your task is to answer the user's question "
        "based *only* on the provided context. If the answer cannot be found within "
        "the context, you must explicitly state 'Information not found in the provided context.' "
        "Do not make up information or use external knowledge."
    )
    human_content = f"Context:\n---\n{rag_context}\n---\n\nQuestion: {question}"
```
- `question`: 用户的原始问题。
- `rag_context`: 这是模拟从知识库中检索到的、与问题相关的上下文信息。在真实的 RAG 系统中，这部分内容会由 `demo_11` (FAISS 检索) 或 `demo_12` (BM25 检索) 的输出来动态填充。
- `system_content`: **系统指令**。这部分内容非常关键，它为 LLM 设定了行为准则。这里的指令要求 LLM：
    - 扮演“乐于助人的助手”角色。
    - **严格依据提供的 `rag_context` 回答问题**。
    - 如果答案不在上下文中，必须明确指出。
    - 禁止编造信息或使用其自身的外部知识。
    这种约束对于保证 RAG 系统答案的真实性和可追溯性至关重要。
- `human_content`: **用户级提示**。这里，我们将 `rag_context` 和 `question` 组合成一个更完整的提示，清晰地呈现给 LLM。使用 `---` 作为分隔符有助于模型区分上下文和实际问题。

#### 2.4. 发送请求给 LLM
```python
    llm_model_name = "gpt-4o-mini" # 选择一个模型
    print(f"\nSending request to OpenAI model: {llm_model_name}...")
    try:
        response = api_processor.send_message(
            model=llm_model_name,
            system_content=system_content,
            human_content=human_content,
            temperature=0.1,
            # max_tokens=100 # 可选参数，限制输出长度
        )
        # ... (打印 LLM 返回的文本答案) ...
    # ... (API 请求过程中的错误处理) ...
```
- `llm_model_name`: 选择要调用的 OpenAI 模型。`gpt-4o-mini` 是一个较新、快速且经济高效的选择。也可以根据需求选择 `gpt-4-turbo` (更强大) 或 `gpt-3.5-turbo` 等。
- `api_processor.send_message(...)`: 这是通过 `APIProcessor` 发送请求的核心调用。参数包括：
    - `model`: 指定模型名称。
    - `system_content`: 我们之前定义的系统指令。
    - `human_content`: 包含上下文和问题的用户级提示。
    - `temperature=0.1`: 设置较低的温度值，使 LLM 的输出更具确定性、更少随机性，这对于基于事实的问答通常是期望的。
    - `max_tokens` (注释掉了): 可以用来限制模型生成答案的最大 token 数量。
- `response`: `send_message` 方法返回的是 LLM 生成的**文本答案**。

#### 2.5. 检查 API 响应元数据
```python
        print("\n--- API Response Metadata (from api_processor.processor.response_data) ---")
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
            # ...
        else:
            print("  No additional response data found on api_processor.processor.")
```
- `APIProcessor` (或者其内部特定于提供商的处理器，如 `OpenAIProcessor`) 在调用 API 后，通常会将其获取到的完整原始响应对象存储起来，例如存储在 `api_processor.processor.response_data` 属性中。
- 这个原始响应对象（对于 OpenAI API，通常是 `openai.types.chat.chat_completion.ChatCompletion` 对象）包含了比最终文本答案更多的信息。
- **`response_metadata.model`**: 确认 API 实际使用了哪个模型版本。
- **`response_metadata.usage`**: 这是一个非常重要的对象，它包含了 token 使用的详细信息：
    - `prompt_tokens`: 输入给模型的提示（包括系统内容和用户内容）所包含的 token 数量。
    - `completion_tokens`: 模型生成的回复内容所包含的 token 数量。
    - `total_tokens`: 上述两者之和，是本次 API 调用总共处理的 token 数量。
- 监控 token 使用量对于估算成本和确保不超过模型的上下文长度限制（不同模型有不同的最大 token 数限制）非常关键。

## 关键启示

1.  **Prompt 的重要性**: 如何构建 `system_content` 和 `human_content` 直接影响 LLM 的回答质量和行为。在 RAG 中，通过 `system_content` 强调“仅依赖上下文”是减少幻觉的关键。
2.  **`APIProcessor` 的抽象作用**: 它可以封装与不同 LLM API 提供商交互的细节，使得上层代码更简洁。
3.  **理解 Token 和成本**: LLM API 的使用成本直接与处理的 token 数量挂钩。学会查看和理解 `usage` 元数据对于实际应用部署至关重要。
4.  **模型选择**: 不同的 LLM 模型有不同的能力、速度和价格。需要根据具体任务需求来选择合适的模型。

## 如何运行脚本

1.  **设置 `OPENAI_API_KEY`**:
    - 在你的项目根目录下创建一个名为 `.env` 的文本文件。
    - 在 `.env` 文件中添加一行：`OPENAI_API_KEY=sk-YourActualOpenAIKeyHere` (将 `sk-YourActualOpenAIKeyHere` 替换为你真实的 OpenAI API 密钥)。
2.  **确保相关库已安装**: `pip install openai python-dotenv`。
3.  **打开终端或命令行工具**。
4.  **导航到脚本所在的目录** (即 `study/` 目录，或者包含 `study` 目录的项目根目录)。
5.  **执行脚本**:
    ```bash
    python study/demo_14_openai_api_request.py
    ```
    或者，如果你在项目的根目录下：
    ```bash
    python -m study.demo_14_openai_api_request
    ```
    脚本将会：
    - 初始化 `APIProcessor`。
    - 使用预设的问题和上下文，构建一个完整的提示。
    - 调用 OpenAI API (例如 `gpt-4o-mini` 模型)。
    - 打印出 LLM 返回的答案。
    - 打印出 API 响应中的元数据，特别是 token 使用情况。

## 总结

`demo_14_openai_api_request.py` 脚本为我们实际演示了如何将检索到的信息（`rag_context`）与用户问题结合，通过一个精心构造的 Prompt 发送给 OpenAI LLM，并获取回答。这构成了 RAG 流程中“生成”环节的基础。

虽然这个例子中的上下文是手动提供的，但在一个完整的 RAG 系统中，这些上下文将由 `demo_11` 或 `demo_12` 中演示的检索模块动态提供。此外，更复杂的 RAG 系统还会使用 `demo_13` 中看到的那些结构化 Prompt 和 Pydantic 模型来处理更复杂的问答场景和获取结构化的 LLM 输出。

希望本教程能帮助你理解与 LLM API 交互的核心步骤！这是构建任何基于 LLM 的智能应用（包括 RAG）的必备技能。
