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
