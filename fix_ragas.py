"""
RAGAS Version0.4.3, which is the last vesion as of 2026-09-29,
has a bug in the imports which causes `import ragas` to fail.
This script fixes the imports to point at langchain_google_vertexai.

Run it with:
    python fix_ragas.py
"""

import importlib.util
from pathlib import Path

REPLACEMENTS = {
    "from langchain_community.chat_models.vertexai import ChatVertexAI":
        "from langchain_google_vertexai import ChatVertexAI",
    "from langchain_community.llms import VertexAI":
        "from langchain_google_vertexai import VertexAI",
}


def main() -> None:
    spec = importlib.util.find_spec("ragas")
    if spec is None or not spec.origin:
        raise SystemExit("ragas is not installed in this environment")

    path = Path(spec.origin).parent / "llms" / "base.py"
    text = path.read_text()
    updated = text
    for old, new in REPLACEMENTS.items():
        updated = updated.replace(old, new)

    if updated == text:
        if all(new in text for new in REPLACEMENTS.values()):
            print(f"already patched: {path}")
            return
        raise SystemExit(f"expected imports not found in {path}")

    path.write_text(updated)
    print(f"patched: {path}")


if __name__ == "__main__":
    main()
