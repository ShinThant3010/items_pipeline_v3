from __future__ import annotations

import json
from pathlib import Path
import sys

import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functions.utils.config import load_config


def main() -> None:
    config = load_config()
    text = "data science career path for electrical engineering students"
    model_name = "gemini-embedding-001"
    dimension = 768

    vertexai.init(project=config.project_id, location=config.region)
    model = TextEmbeddingModel.from_pretrained(model_name)
    embedding = model.get_embeddings(
        [TextEmbeddingInput(text=text, task_type="RETRIEVAL_QUERY")],
        output_dimensionality=dimension,
    )[0]

    record = {
        "text": text,
        "embedding": embedding.values,
        "model": model_name,
        "dimension": dimension,
    }

    output_path = Path("tests/embed_text/embeddings.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
