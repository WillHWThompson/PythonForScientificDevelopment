#!/bin/bash

# setup_repo.sh
# Automates the setup of a Scientific Python + LLM repository using `uv`.
# Usage: ./setup_repo.sh [project_name] [python_version]

set -e

PROJECT_NAME=${1:-"."}
PYTHON_VERSION=${2:-"3.12"}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "🚀 Starting Scientific Python Repo Setup..."

# 1. Check for uv
if ! command_exists uv; then
    echo "❌ 'uv' is not installed. Installing it now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source source to make uv available in current session if needed
    source "$HOME/.cargo/env" 2>/dev/null || true
else
    echo "✅ 'uv' found."
fi

# 2. Initialize Project
# We use --lib to ensure it's configured as an installable Python library (src layout)
if [ "$PROJECT_NAME" != "." ]; then
    echo "📂 Creating project directory: $PROJECT_NAME"
    uv init --lib "$PROJECT_NAME" --python "$PYTHON_VERSION"
    cd "$PROJECT_NAME"
else
    echo "📂 Initializing in current directory..."
    # If running in existing dir, we assume we want to turn it into a library
    uv init --lib --python "$PYTHON_VERSION"
fi

# 3. Add Dependencies
echo "📦 Installing Scientific Stack..."
uv add numpy pandas scipy matplotlib seaborn jupyterlab ipykernel jax jaxlib

echo "🤖 Installing LLM & Dev Stack..."
uv add python-dotenv pydantic openai anthropic

echo "🛠️ Installing Dev Tools..."
uv add --dev ruff pytest

# 4. Standardize Directory Structure
echo "txCreating standard directory structure..."
mkdir -p notebooks tests data/raw data/processed models

# Note: 'uv init --lib' automatically creates src/<package_name>/__init__.py

# 5. Create .env.example
if [ ! -f ".env.example" ]; then
    echo "📝 Creating .env.example..."
    cat > .env.example <<EOL
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...

# Config
DEBUG=True
EOL
fi

# 6. Setup Git ignore extension
if [ -f ".gitignore" ]; then
    # Ensure .env is ignored
    if ! grep -q ".env" .gitignore; then
        echo ".env" >> .gitignore
    fi
    # Ensure data is ignored (optional, but good practice for large data)
    if ! grep -q "data/" .gitignore; then
        echo "data/" >> .gitignore
    fi
fi

# 7. Sync to ensure environment is ready and package is installed
echo "🔄 Syncing environment..."
uv sync

# 8. Register Jupyter Kernel
echo "🌽 Registering Jupyter Kernel..."
# We extract a clean kernel name from the project name
KERNEL_NAME=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]' | tr -cd '[a-z0-9_]')
if [ "$KERNEL_NAME" = "." ]; then
    KERNEL_NAME=$(basename "$PWD" | tr '[:upper:]' '[:lower:]' | tr -cd '[a-z0-9_]')
fi

uv run python -m ipykernel install --user --name="$KERNEL_NAME" --display-name "Python ($KERNEL_NAME)"

echo "✅ Setup Complete!"
echo "------------------------------------------------"
echo "To get started:"
if [ "$PROJECT_NAME" != "." ]; then
    echo "  cd $PROJECT_NAME"
fi
echo "  uv run jupyter lab"
echo "------------------------------------------------"
