import pathlib

import yaml

_CORPUS_DIR = pathlib.Path(__file__).parent / "corpus"


def load_case(case_id: str) -> dict:
    path = _CORPUS_DIR / f"{case_id}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
