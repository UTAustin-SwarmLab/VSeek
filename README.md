Next, install uv:
```bash
pip install uv
```

Next, set up venv:
```bash
uv venv .venv
```


Finally, install everything in `pyproject.toml` to build project dependencies:
```bash
uv sync
```

