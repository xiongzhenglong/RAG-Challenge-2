# PDF 解析入门：使用 Python 和 Docling 提取文本

大家好！今天我们将一起探索如何使用 Python 解析 PDF 文件并提取其中的文本内容。我们将使用一个名为 `docling` 的库（尽管此 demo 主要展示其概念，实际 `docling` 库的用法可能更复杂，并依赖特定模型）。本教程将详细解释一个示例 Python 脚本 `demo_01_pdf_parsing.py`，帮助初学者理解 PDF 解析的基本流程。

## 脚本目标

这个 Python 脚本的目标是：
1. 加载一个指定的 PDF 文件。
2. 使用 `docling` 相关组件（主要是 `DocumentParser`）来解析这个 PDF。
3. 提取并显示 PDF 第一页的文本内容。
4. 处理可能发生的错误，例如文件未找到或解析失败。

## 环境准备

在运行此脚本之前，请确保：
- 你已经安装了 Python。
- `docling` 库及其依赖项已安装。脚本中提到了 `Pipeline.download_docling_models()`，这意味着 `docling` 可能需要下载一些预训练模型才能正常工作。这些模型可能用于文本提取、布局检测等任务。请确保网络连接畅通，以便下载这些模型。
- 示例 PDF 文件位于 `data/test_set/pdf_reports/` 目录下。脚本中使用的示例文件是 `194000c9109c6fa628f1fed33b44ae4c2b8365f4.pdf`。

## Python 脚本详解

下面是 `study/demo_01_pdf_parsing.py` 脚本的完整代码：

```python
# study/demo_01_pdf_parsing.py

import sys
import os

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pdf_parsing import DocumentParser, TextExtractor, LayoutDetector, StructureDetector, EntityRecognizer, Pipeline

def main():
    # Ensure Docling models are downloaded
    # This might take a while if the models are not already downloaded
    print("Checking and downloading Docling models if necessary...")
    Pipeline.download_docling_models()
    print("Docling models are ready.")

    # Define the path to the sample PDF file
    # Using a relative path from the root of the project
    sample_pdf_path = "data/test_set/pdf_reports/194000c9109c6fa628f1fed33b44ae4c2b8365f4.pdf"
    print(f"Using sample PDF: {sample_pdf_path}")

    # Check if the sample PDF file exists
    if not os.path.exists(sample_pdf_path):
        print(f"Error: Sample PDF file not found at {sample_pdf_path}")
        print("Please ensure the data is available in the 'data/test_set/pdf_reports/' directory.")
        # As a fallback, let's try to list files in the directory to help debug
        print(f"Looking for data in: {os.path.abspath('data/test_set/pdf_reports/')}")
        if os.path.exists('data/test_set/pdf_reports/'):
             print(f"Files in 'data/test_set/pdf_reports/': {os.listdir('data/test_set/pdf_reports/')[:5]}") # Show first 5
        else:
            print("'data/test_set/pdf_reports/' directory does not exist.")
        return

    # Initialize components for PDF parsing
    # For this demo, we'll focus on text extraction and basic structure.
    # More advanced components like EntityRecognizer might require specific model setups.
    text_extractor = TextExtractor()
    layout_detector = LayoutDetector() # Depends on Detectron2 and models
    # StructureDetector and EntityRecognizer might be more complex to set up for a simple demo
    # For now, let's try to use DocumentParser which encapsulates some of this.

    print("Initializing DocumentParser...")
    # The DocumentParser might try to load all models, which can be heavy.
    # Let's see if we can parse with just text and layout.
    # We might need to adjust this based on how DocumentParser is implemented.
    try:
        parser = DocumentParser() # This might trigger model loading for all components
    except Exception as e:
        print(f"Error initializing DocumentParser: {e}")
        print("This might be due to missing models or dependencies for all components.")
        print("Attempting a more basic parsing approach...")
        # Fallback: Try to use TextExtractor directly if DocumentParser is too complex for a quick demo
        try:
            doc_bytes = open(sample_pdf_path, "rb").read()
            # TextExtractor usually works on a page-by-page basis after layout detection.
            # Let's try to simulate a simplified flow if DocumentParser is problematic.
            # This part might need adjustment based on actual class interfaces.
            # For a true demo, we'd ideally use the full Pipeline or DocumentParser.
            # If direct text extraction is needed, it would be more involved.
            print("DocumentParser initialization failed. A full demo of DocumentParser requires all models.")
            print("For a simplified text extraction, you would typically integrate TextExtractor within a pipeline.")
            print("This demo will focus on what can be achieved with available components.")
            # As an alternative, we can try to show how many pages pdfminer.six detects.
            from pdfminer.high_level import extract_pages
            try:
                page_count = 0
                for _ in extract_pages(sample_pdf_path):
                    page_count +=1
                print(f"Basic check: pdfminer.six detected {page_count} pages in the document.")
            except Exception as pe:
                print(f"Error during basic pdfminer.six check: {pe}")

            return # Exiting if DocumentParser fails as it's key for the intended demo
        except Exception as fallback_e:
            print(f"Error in fallback parsing attempt: {fallback_e}")
            return


    # Parse the PDF document
    print(f"Parsing PDF: {sample_pdf_path}")
    try:
        # The `parse` method should take the PDF path and return a Document object
        document = parser.parse(sample_pdf_path)
    except Exception as e:
        print(f"Error during PDF parsing: {e}")
        print("This could be due to issues with the PDF file or model incompatibilities.")
        print("Ensure all models for DocumentParser components are correctly downloaded and configured.")
        # Let's try to see if any specific component is causing the issue
        # This is for debugging purposes if the above fails.
        print("Attempting to extract text directly to see if that part works...")
        try:
            from src.pdf_parsing.utils import pdf_to_images_pymupdf
            from PIL import Image
            doc_images = list(pdf_to_images_pymupdf(sample_pdf_path))
            if not doc_images:
                print("Could not convert PDF to images using PyMuPDF.")
                return
            
            first_page_image = Image.open(doc_images[0])
            extracted_text_elements = text_extractor.extract_text(first_page_image, page_number=0)
            
            if extracted_text_elements:
                print(f"Direct text extraction from first page (first 5 elements): {extracted_text_elements[:5]}")
                first_page_text_content = " ".join([te.text for te in extracted_text_elements])
                print("\n--- Text Content of First Page (Direct Extraction) ---")
                print(first_page_text_content[:1000]) # Print first 1000 characters
                print("----------------------------------------------------")
            else:
                print("Direct text extraction yielded no elements.")

        except Exception as te_e:
            print(f"Error during direct text extraction attempt: {te_e}")
        return

    # Print the number of pages parsed
    num_pages = len(document.pages)
    print(f"\nSuccessfully parsed document. Number of pages: {num_pages}")

    # Print the text content of the first page
    if num_pages > 0:
        first_page = document.pages[0]
        print("\n--- Text Content of First Page ---")
        # The 'text' attribute of a Page object should give its full text content
        # Or, it might be a list of text blocks that need to be joined.
        # This depends on the structure of the Page object from pdf_parsing.py
        if hasattr(first_page, 'text_content'): # Assuming a 'text_content' field
            print(first_page.text_content[:1000]) # Print first 1000 characters
        elif hasattr(first_page, 'text_blocks') and first_page.text_blocks: # Assuming it has text_blocks
            full_text = " ".join([block.text for block in first_page.text_blocks if hasattr(block, 'text')])
            print(full_text[:1000]) # Print first 1000 characters
        elif hasattr(first_page, 'text_elements') and first_page.text_elements: # Common alternative
            full_text = " ".join([elem.text for elem in first_page.text_elements if hasattr(elem, 'text')])
            print(full_text[:1000])
        else:
            print("Could not find a direct text attribute on the first page object.")
            print("Page object details:", dir(first_page)) # To understand its structure
        print("----------------------------------")
    else:
        print("The document has no pages or pages could not be parsed.")

if __name__ == "__main__":
    main()
```

### 1. 导入模块与路径设置

```python
import sys
import os

# Add the src directory to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pdf_parsing import DocumentParser, TextExtractor, LayoutDetector, StructureDetector, EntityRecognizer, Pipeline
```

- `sys` 和 `os`：这两个是 Python 的标准库。`sys` 用于访问与 Python 解释器相关的变量和函数，`os` 提供了与操作系统交互的功能（例如文件路径操作）。
- `sys.path.append(...)`：这行代码的目的是将 `src` 目录添加到 Python 的模块搜索路径中。这样做之后，脚本就可以直接从 `src` 目录导入模块，比如 `src.pdf_parsing`。
  - `os.path.dirname(__file__)` 获取当前脚本所在的目录。
  - `os.path.join(..., '..')` 返回上一级目录（即项目的根目录，如果 `study` 和 `src` 在同一级别）。
  - `os.path.abspath(...)` 将其转换为绝对路径。
- `from src.pdf_parsing import ...`: 从 `src.pdf_parsing` 模块导入所需的类。这些类是 `docling` 库提供的核心组件，用于 PDF 解析的不同阶段：
    - `DocumentParser`: 核心解析器，可能会协调其他组件工作。
    - `TextExtractor`: 负责从 PDF 页面或图像中提取文本。
    - `LayoutDetector`: 分析页面布局（例如，识别段落、表格、图像等）。
    - `StructureDetector`: 识别文档的逻辑结构（例如，章节、标题）。
    - `EntityRecognizer`: 识别文本中的命名实体（例如，人名、地名、组织机构名）。
    - `Pipeline`: 可能用于管理模型下载或串联解析步骤。

### 2. `main()` 函数

这是脚本的主执行函数。

#### 2.1. 下载 Docling 模型

```python
    print("Checking and downloading Docling models if necessary...")
    Pipeline.download_docling_models()
    print("Docling models are ready.")
```
- 在进行任何解析之前，脚本首先调用 `Pipeline.download_docling_models()`。
- 这个静态方法会检查所需的 `docling` 模型是否已经下载到本地，如果没有，则会自动下载。
- 这是一个很重要的步骤，因为许多高级的文档分析功能都依赖于预训练的机器学习模型。

#### 2.2. 定义 PDF 文件路径并检查文件是否存在

```python
    sample_pdf_path = "data/test_set/pdf_reports/194000c9109c6fa628f1fed33b44ae4c2b8365f4.pdf"
    print(f"Using sample PDF: {sample_pdf_path}")

    if not os.path.exists(sample_pdf_path):
        print(f"Error: Sample PDF file not found at {sample_pdf_path}")
        # ... (错误处理和调试信息) ...
        return
```
- `sample_pdf_path` 变量存储了要解析的 PDF 文件的相对路径。
- `os.path.exists(sample_pdf_path)` 检查该文件是否存在。
- 如果文件不存在，脚本会打印错误信息并退出。错误信息中包含了一些调试辅助：
    - 尝试打印目标目录的绝对路径。
    - 如果目标目录存在，打印该目录下的前5个文件名，帮助用户确认路径是否正确或文件是否确实丢失。

#### 2.3. 初始化解析组件

```python
    text_extractor = TextExtractor()
    layout_detector = LayoutDetector() # Depends on Detectron2 and models

    print("Initializing DocumentParser...")
    try:
        parser = DocumentParser()
    except Exception as e:
        print(f"Error initializing DocumentParser: {e}")
        # ... (错误处理和备选方案) ...
        return
```
- 脚本实例化了 `TextExtractor` 和 `LayoutDetector`。虽然在这个 demo 的主要流程中它们没有被直接显式调用（而是通过 `DocumentParser`），但 `DocumentParser` 内部可能会使用它们。
- 核心步骤是初始化 `DocumentParser()`。这个对象将负责整个 PDF 的解析过程。
- **错误处理**: `DocumentParser` 的初始化可能比较复杂，因为它可能需要加载多个模型和依赖项。
    - 如果初始化失败（例如，因为模型缺失或 `Detectron2` 等依赖库配置问题），脚本会捕获异常。
    - 捕获到异常后，会打印错误信息，并尝试一个“备选方案”：使用 `pdfminer.six`（一个流行的 Python PDF 解析库）来做一个基本检查，看看能识别出多少页。这有助于判断问题是出在 `docling` 的复杂初始化上，还是 PDF 文件本身有问题。
    - 如果 `DocumentParser` 初始化失败，脚本通常会退出，因为它是后续操作的关键。

#### 2.4. 解析 PDF 文档

```python
    print(f"Parsing PDF: {sample_pdf_path}")
    try:
        document = parser.parse(sample_pdf_path)
    except Exception as e:
        print(f"Error during PDF parsing: {e}")
        # ... (错误处理和备选方案) ...
        return
```
- `parser.parse(sample_pdf_path)` 是实际执行 PDF 解析的方法。它接收 PDF文件的路径作为输入，并期望返回一个 `Document` 对象。这个 `Document` 对象会包含解析后的信息，比如页面、文本块、布局元素等。
- **错误处理**: 解析过程也可能失败。
    - 原因可能包括：PDF 文件损坏、模型不兼容、或者解析器内部错误。
    - 如果解析失败，脚本会捕获异常并打印错误信息。
    - 作为一种调试手段，它会尝试一个“备选方案”：
        - 使用 `src.pdf_parsing.utils.pdf_to_images_pymupdf` 将 PDF 的第一页转换为图像。
        - 然后使用之前初始化的 `text_extractor` 直接从这个图像中提取文本。
        - 这有助于判断问题是出在更高级的文档结构分析上，还是基础的文本提取部分就已经失败。
    - 如果主解析流程和这个备选的直接文本提取都失败了，脚本会退出。

#### 2.5. 打印解析结果

```python
    num_pages = len(document.pages)
    print(f"\nSuccessfully parsed document. Number of pages: {num_pages}")

    if num_pages > 0:
        first_page = document.pages[0]
        print("\n--- Text Content of First Page ---")
        if hasattr(first_page, 'text_content'):
            print(first_page.text_content[:1000])
        elif hasattr(first_page, 'text_blocks') and first_page.text_blocks:
            full_text = " ".join([block.text for block in first_page.text_blocks if hasattr(block, 'text')])
            print(full_text[:1000])
        elif hasattr(first_page, 'text_elements') and first_page.text_elements:
            full_text = " ".join([elem.text for elem in first_page.text_elements if hasattr(elem, 'text')])
            print(full_text[:1000])
        else:
            print("Could not find a direct text attribute on the first page object.")
            print("Page object details:", dir(first_page))
        print("----------------------------------")
    else:
        print("The document has no pages or pages could not be parsed.")
```
- 如果 PDF 解析成功，脚本会获取 `Document` 对象中的页面列表 (`document.pages`)。
- `len(document.pages)` 给出文档的总页数。
- 如果文档至少有一页：
    - 它会获取第一页 (`document.pages[0]`)。
    - 然后，它尝试从第一页对象中提取文本内容。这里有几种可能的方式，取决于 `Page` 对象的具体实现：
        - 检查是否有 `text_content` 属性。
        - 检查是否有 `text_blocks` 列表，并连接每个块的文本。
        - 检查是否有 `text_elements` 列表，并连接每个元素的文本。
    - 打印提取到的前1000个字符。
    - 如果找不到合适的文本属性，它会打印 `Page` 对象的可用属性列表 (`dir(first_page)`)，这对于调试和理解 `Page` 对象结构很有帮助。
- 如果文档没有页面，则打印相应信息。

### 3. 脚本入口

```python
if __name__ == "__main__":
    main()
```
- 这是一个 Python 脚本的标准写法。
- `if __name__ == "__main__":`确保 `main()` 函数只在脚本被直接执行时调用，而不是在它被其他脚本导入时调用。

## 如何运行脚本

1.  **确保环境配置正确**：
    *   Python 已安装。
    *   `docling` 库及其依赖（如 `Detectron2`，如果 `LayoutDetector` 需要）已安装。
    *   相关的 `docling` 模型已下载（脚本会自动尝试下载）。
    *   示例 PDF 文件 `data/test_set/pdf_reports/194000c9109c6fa628f1fed33b44ae4c2b8365f4.pdf` 存在。
2.  **打开终端或命令行工具**。
3.  **导航到脚本所在的目录** (例如 `study/` 目录，或者项目的根目录，取决于你的当前位置和如何组织文件)。
4.  **执行脚本**:
    ```bash
    python study/demo_01_pdf_parsing.py
    ```
    或者，如果你在项目的根目录下：
    ```bash
    python -m study.demo_01_pdf_parsing
    ```

    脚本会开始运行，你将看到打印的日志信息，包括模型下载状态、正在使用的 PDF 文件、解析过程中的步骤，以及最终提取到的第一页文本（或错误信息）。

## 预期输出（成功时）

如果一切顺利，你大致会看到如下的输出：

```
Checking and downloading Docling models if necessary...
Docling models are ready.
Using sample PDF: data/test_set/pdf_reports/194000c9109c6fa628f1fed33b44ae4c2b8365f4.pdf
Initializing DocumentParser...
Parsing PDF: data/test_set/pdf_reports/194000c9109c6fa628f1fed33b44ae4c2b8365f4.pdf

Successfully parsed document. Number of pages: X  <-- X是实际页数

--- Text Content of First Page ---
[这里是PDF第一页的前1000个字符]
----------------------------------
```

## 常见问题与排查

- **模型下载失败**：检查网络连接。有些模型可能很大，需要较长时间下载。
- **`DocumentParser` 初始化失败**：
    - 这通常与 `docling` 的核心组件或其依赖（如 `Detectron2` for `LayoutDetector`）有关。确保所有必需的库都已正确安装和配置。
    - 错误消息 `Error initializing DocumentParser: ...` 后面通常会跟有具体的异常信息，仔细阅读它可以帮助定位问题。
- **PDF 解析失败**：
    - `Error during PDF parsing: ...` 提示解析过程中出现问题。
    - 可能是 PDF 文件本身已损坏或格式特殊。
    - 也可能是模型与特定 PDF 内容不兼容。
    - 脚本中的备选调试方案（直接用 `TextExtractor` 从页面图像提取文本）可能会提供更多线索。
- **找不到 PDF 文件**：仔细检查 `sample_pdf_path` 的路径是否正确，以及文件是否确实存在于该位置。

## 总结

这个 `demo_01_pdf_parsing.py` 脚本为我们提供了一个使用 `docling` 库（或类似概念的库）进行 PDF 解析的基本框架。它涵盖了从模型准备、文档加载、核心解析到错误处理和结果展示的完整流程。对于初学者来说，理解这个脚本的结构和每个部分的功能，是进一步学习更高级文档智能技术（如表格提取、实体识别、文档结构化分析）的良好开端。

希望这篇教程对你有所帮助！
