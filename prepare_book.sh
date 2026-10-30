
#!/usr/bin/env bash
set -e

echo "=== Step 1: Convert ==="
uv run python convert.py

echo "=== Step 2: Clean ==="
uv run python cleaner.py

echo "=== Step 3: Embedding ==="
uv run python embedding.py

echo "=== Pipeline completed successfully ==="
