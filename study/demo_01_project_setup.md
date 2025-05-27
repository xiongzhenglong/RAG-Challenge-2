# Python 项目环境搭建指南

大家好！在开始任何 Python 项目之前，正确地搭建开发环境是至关重要的一步。一个良好的开端能确保项目顺利进行，避免许多后续可能出现的问题。本教程将引导你完成一个典型的 Python 项目（尤其是类似我们当前讨论的 AI 或数据处理相关的项目）所需的基础环境配置步骤。

这些步骤是根据一个名为 `study/demo_01_project_setup.py` 的指导脚本整理而来的。该脚本本身并不执行复杂的任务，而是打印出一系列设置指令。

## 为何需要项目环境设置？

- **依赖隔离**: 不同项目可能需要不同版本的库。虚拟环境可以防止版本冲突。
- **复现性**: 确保其他开发者或部署环境能够以相同配置运行项目。
- **敏感信息管理**: API 密钥等敏感数据需要安全存储，而不是硬编码到代码中。

## `demo_01_project_setup.py` 脚本内容

首先，让我们看看这个 Python 脚本 (`study/demo_01_project_setup.py`) 的内容，它概述了我们需要遵循的步骤：

```python
# study/demo_01_project_setup.py

"""
This script provides instructions for setting up the project environment.
It covers creating a virtual environment, installing dependencies,
and configuring API keys.
"""

print("Starting project setup instructions...")

# --- Step 1: Virtual Environment ---
print("\n--- Step 1: Create a Virtual Environment ---")
print("A virtual environment is recommended to manage project-specific dependencies.")
print("It isolates your project's packages from the global Python installation.")
print("\nTo create a virtual environment (if you haven't already):")
print("1. Open your terminal or command prompt.")
print("2. Navigate to the root directory of this project.")
print("3. Run the following command:")
print("   python -m venv venv")
print("   (This will create a folder named 'venv' in your project directory)")

# --- Step 2: Activate Virtual Environment ---
print("\n--- Step 2: Activate the Virtual Environment ---")
print("Before installing dependencies, you need to activate the virtual environment.")
print("\nActivation commands differ based on your operating system and shell:")
print("- For Windows PowerShell:")
print("  .\\venv\\Scripts\\Activate.ps1")
print("- For Windows Command Prompt (cmd.exe):")
print("  venv\\Scripts\\activate.bat")
print("- For Unix-like systems (Linux, macOS, Git Bash on Windows):")
print("  source venv/bin/activate")
print("\nAfter activation, your terminal prompt should usually change to indicate")
print("that the virtual environment is active (e.g., '(venv)' at the beginning).")

# --- Step 3: Install Dependencies ---
print("\n--- Step 3: Install Dependencies ---")
print("Once the virtual environment is active, install the required packages.")
print("This project uses 'pip' for package management and lists dependencies in")
print("'requirements.txt' and setup files.")
print("\nRun the following command in your activated environment:")
print("   pip install -e . -r requirements.txt")
print("   - '-e .' installs the current project in editable mode.")
print("   - '-r requirements.txt' installs packages listed in requirements.txt.")

# --- Step 4: Configure Environment Variables (.env file) ---
print("\n--- Step 4: Configure Environment Variables (.env file) ---")
print("This project uses a '.env' file to store sensitive information like API keys.")
print("You should have a file named 'env' (or 'env.example') in the project root.")
print("\n1. Rename 'env' to '.env':")
print("   - If you see a file named 'env', rename it to '.env'.")
print("   - If you see 'env.example', copy it to '.env'.")
print("   (Files starting with a dot might be hidden by default in some file explorers.)")

# --- Step 5: Add API Keys to .env ---
print("\n--- Step 5: Add Your API Keys to the .env file ---")
print("Open the '.env' file with a text editor and add your API keys.")
print("Below are placeholders. Replace 'YOUR_API_KEY_HERE' with your actual keys.")
print("\nExample content for .env:")
print("# OpenAI API Key")
print("OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE")
print("\n# Jina API Key (Optional, if you use Jina for embeddings)")
print("# JINA_API_KEY=YOUR_JINA_API_KEY_HERE")
print("\n# IBM Cloud API Key for WatsonX (Optional)")
print("# IBM_CLOUD_API_KEY=YOUR_IBM_CLOUD_API_KEY_HERE")
print("# IBM_CLOUD_URL=YOUR_IBM_CLOUD_URL_HERE")
print("# IBM_PROJECT_ID=YOUR_IBM_PROJECT_ID_HERE")
print("\n# Google Gemini API Key (Optional)")
print("# GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE")
print("\nSave the .env file after adding your keys.")
print("IMPORTANT: Do NOT commit the .env file to version control (e.g., Git).")
print("The '.gitignore' file should already be configured to ignore '.env'.")

# --- Completion ---
print("\n--- Setup Complete ---")
print("Project setup instructions complete. Please ensure you have followed all steps.")
print("If you encounter any issues, review the steps or consult the project documentation.")
```

## 详细设置步骤

现在，我们将按照脚本中的指示，一步步完成项目环境的搭建。

### 第 1 步：创建虚拟环境

**为什么需要虚拟环境？**
虚拟环境为你的项目创建一个独立、隔离的 Python 运行环境。这意味着项目安装的库和依赖项将与你系统全局的 Python 环境以及其他项目的环境分开，避免了版本冲突和混乱。

**如何创建？**
1.  打开你的终端（在 Windows 上可能是命令提示符 CMD 或 PowerShell，在 Linux 或 macOS 上是 Terminal）。
2.  使用 `cd` 命令导航到你的项目根目录。
3.  运行以下命令：
    ```bash
    python -m venv venv
    ```
    或者，如果你同时安装了 Python 2 和 Python 3，可能需要明确使用 `python3`:
    ```bash
    python3 -m venv venv
    ```
    这个命令会在你的项目根目录下创建一个名为 `venv` 的文件夹，其中包含了 Python 解释器的一个副本以及管理包的工具。

### 第 2 步：激活虚拟环境

创建虚拟环境后，你需要激活它才能开始使用。激活后，你安装的任何包都将安装到这个特定的虚拟环境中，而不是全局 Python 环境。

**如何激活？**
激活命令因操作系统和使用的终端而异：

*   **Windows PowerShell**:
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```
    (如果遇到执行策略问题，你可能需要先运行 `Set-ExecutionPolicy Unrestricted -Scope Process` 或 `Set-ExecutionPolicy RemoteSigned -Scope Process`)

*   **Windows 命令提示符 (cmd.exe)**:
    ```batch
    venv\Scripts\activate.bat
    ```

*   **Unix 或类 Unix 系统 (Linux, macOS, Windows 上的 Git Bash)**:
    ```bash
    source venv/bin/activate
    ```

成功激活后，通常你的终端提示符前面会出现 `(venv)` 字样，表明当前虚拟环境已激活。

### 第 3 步：安装依赖项

项目通常依赖于多个外部 Python 包。这些依赖项通常记录在 `requirements.txt` 文件中，并且项目本身可能也需要作为包安装（特别是如果它提供了命令行工具或可导入模块）。

**如何安装？**
确保你的虚拟环境已激活，然后在项目根目录下运行：
```bash
pip install -e . -r requirements.txt
```
让我们分解这个命令：
- `pip install`: 这是 Python 的包安装命令。
- `-e .`:
    - `.` 指的是当前目录（即你的项目根目录）。
    - `-e` 是 `--editable` 的缩写。它以“可编辑”模式安装当前目录下的项目。这意味着如果你修改了项目的源代码，更改会立即生效，无需重新安装。这对于开发非常方便。这通常要求你的项目根目录下有一个 `setup.py` 或 `pyproject.toml` 文件来定义项目如何被打包。
- `-r requirements.txt`:
    - `-r` 是 `--requirement` 的缩写。
    - `requirements.txt` 是一个文本文件，其中列出了项目运行所需的所有第三方库及其版本。`pip` 会读取这个文件并安装所有列出的包。

### 第 4 步：配置环境变量 (`.env` 文件)

为了安全地管理敏感信息（如 API 密钥、数据库密码等），而不是将它们硬编码到代码中，通常会使用 `.env` (dotenv) 文件。

**如何配置？**
1.  在你的项目根目录下，查找名为 `env` 或 `env.example` 的文件。
    *   如果找到 `env` 文件，将其重命名为 `.env`。
    *   如果找到 `env.example` 文件，复制一份并将其命名为 `.env`。
    (`env.example` 通常是一个模板，列出了项目需要的环境变量但没有包含实际值。)
2.  **注意**：以点 `.` 开头的文件（如 `.env`）在某些操作系统或文件浏览器中可能是默认隐藏的。你需要调整设置才能看到它们。

    在命令行中，你可以使用以下命令（根据你的系统）来重命名或复制：
    *   重命名 (Linux/macOS/Git Bash): `mv env .env`
    *   复制 (Linux/macOS/Git Bash): `cp env.example .env`
    *   重命名 (Windows CMD): `ren env .env`
    *   复制 (Windows CMD): `copy env.example .env`

### 第 5 步：在 `.env` 文件中添加 API 密钥

打开你刚刚创建或重命名的 `.env` 文件（使用任何文本编辑器，如 VS Code, Notepad++, Sublime Text, Vim 等），然后填入你的实际 API 密钥。

文件中通常会有占位符，你需要将它们替换掉。

**`.env` 文件示例内容：**
```env
# OpenAI API Key
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE

# Jina API Key (Optional, if you use Jina for embeddings)
# JINA_API_KEY=YOUR_JINA_API_KEY_HERE

# IBM Cloud API Key for WatsonX (Optional)
# IBM_CLOUD_API_KEY=YOUR_IBM_CLOUD_API_KEY_HERE
# IBM_CLOUD_URL=YOUR_IBM_CLOUD_URL_HERE
# IBM_PROJECT_ID=YOUR_IBM_PROJECT_ID_HERE

# Google Gemini API Key (Optional)
# GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
```

将 `YOUR_..._API_KEY_HERE` 替换为你的真实密钥。如果某些服务你用不到，可以暂时保留占位符或者注释掉相应的行（在 `.env` 文件中，通常以 `#` 开头的行是注释）。

**非常重要：**
确保 `.env` 文件被添加到了项目的 `.gitignore` 文件中（如果项目使用 Git 进行版本控制）。这可以防止你不小心将包含敏感密钥的 `.env` 文件提交到代码仓库（如 GitHub），从而避免密钥泄露。通常，标准的 Python 项目的 `.gitignore` 文件会包含 `*.env` 或直接 `.env`。

## 设置完成

恭喜！完成以上所有步骤后，你的项目开发环境就基本搭建好了。现在你应该能够顺利运行项目代码、进行开发和测试了。

如果在设置过程中遇到任何问题，请仔细回顾每个步骤，检查命令是否输入正确，或者查阅相关工具的官方文档。

祝你编码愉快！
