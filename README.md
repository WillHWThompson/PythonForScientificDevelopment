# Scientific Python Development with `uv` and LLMs

This repository serves as a guide and template for setting up a modern, high-performance Python environment for scientific computing and LLM-based development.

We use **`uv`** instead of Conda because it is:
- 🚀 **Extremely Fast**: Resolves dependencies in milliseconds.
- 📦 **Standardized**: Uses standard `pyproject.toml` and lockfiles for reproducible builds.
- 🛠️ **Unified**: Manages Python versions, virtual environments, and dependencies in one tool.

## 🚀 Quick Start (Automated)

We provide a script to go from zero to a fully configured environment.

1.  **Clone this repo** (or download the script).
2.  **Make the script executable**:
    ```bash
    chmod +x setup_repo.sh
    ```
3.  **Run the script**:
    ```bash
    # Usage: ./setup_repo.sh <project_name> <python_version>
    ./setup_repo.sh my_new_project 3.12
    ```
4.  **Enter your project**:
    ```bash
    cd my_new_project
    uv run jupyter lab
    ```

---

## 📚 Step-by-Step Tutorial (Manual)

If you prefer to understand what's happening under the hood (or just love typing), here is the manual setup process.

### 1. Install `uv`

First, install `uv` if you haven't already.

**MacOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Initialize a Project

Create a new directory and initialize it. `uv` will create a `pyproject.toml` and a `.python-version` file for you.

```bash
mkdir my_science_project
cd my_science_project
uv init --python 3.12
```

This creates a lightweight virtual environment at `.venv`. **No more manually creating or activating environments!** `uv` handles it for you.

### 3. Install Scientific Stack

Add the core libraries for data science. `uv add` installs them and updates your `pyproject.toml` automatically.

```bash
uv add numpy pandas scipy matplotlib seaborn
```

### 4. Install Jupyter & LLM Tools

We need Jupyter for notebooks and libraries for working with LLMs (OpenAI, Anthropic) and robust data handling (Pydantic).

```bash
# Notebooks
uv add jupyterlab ipykernel

# LLM Stack
uv add python-dotenv pydantic openai anthropic

### 5. Register Jupyter Kernel

To make this environment available in Jupyter notebooks, register it as a kernel:

```bash
uv run python -m ipykernel install --user --name=my_science_project --display-name "Python (My Science Project)"
```
```

### 5. Install Development Tools

Add tools for code quality (linting/formatting) and testing. We add these as **dev** dependencies so they don't bloat production builds.

```bash
uv add --dev ruff pytest
```

### 6. Project Structure

Organize your code for scalability.

```bash
mkdir -p src notebooks tests data/raw data/processed models
mv hello.py src/  # Move the default file created by init
```

Recommended layout:
```text
my_science_project/
├── data/               # Local data (ignored by git)
├── notebooks/          # Jupyter notebooks
├── src/                # Source code modules
├── tests/              # Unit tests
├── pyproject.toml      # Dependency definition
├── uv.lock             # Exact dependency versions (Reproducibility!)
└── .venv/              # Virtual environment (managed by uv)
```

### 7. Running Code

With `uv`, you don't need to manually activate the environment. Just use `uv run`.

**Run a script:**
```bash
uv run src/hello.py
```

**Run Jupyter Lab:**
```bash
uv run jupyter lab
```

## 📝 Best Practices

1.  **Always use `uv.lock`**: Commit this file. It ensures your collaborators have the *exact* same package versions as you.
2.  **Keep secrets out of Git**: Use `.env` for API keys and add it to `.gitignore`.
3.  **Put logic in `src/`**: Don't write complex functions in Notebooks. Write them in `src/` modules and import them in notebooks.
