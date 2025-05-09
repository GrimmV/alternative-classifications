
# Setup uv for virtual environment

```curl -LsSf https://astral.sh/uv/install.sh | sh```

```
uv python install 3.11
uv venv .venv --python python3.11
source $HOME/.local/bin/env
source .venv/bin/activate
uv pip install pip
```

```
pip install -r requirements.txt
(Or upgrade: pip install --upgrade --force-reinstall -r requirements.txt)
```

# Install Ollama

```curl -fsSL https://ollama.com/install.sh | sh```

# Test

```python llama_model.py```