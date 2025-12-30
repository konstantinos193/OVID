from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Optional

from .config import load_settings


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    pipeline: str
    extra: Dict[str, str]


def _read_model_json(model_dir: Path) -> Optional[ModelSpec]:
    spec_path = model_dir / "model.json"
    if not spec_path.exists():
        return None
    with spec_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    name = data.get("name", model_dir.name)
    pipeline = data.get("pipeline", "diffusers")
    extra = {k: v for k, v in data.items() if k not in {"name", "pipeline"}}
    return ModelSpec(name=name, path=model_dir, pipeline=pipeline, extra=extra)


def discover_models(models_dir: Path) -> Dict[str, ModelSpec]:
    if not models_dir.exists():
        return {}
    models: Dict[str, ModelSpec] = {}
    for child in models_dir.iterdir():
        if not child.is_dir():
            continue
        spec = _read_model_json(child)
        if spec:
            models[spec.name] = spec
    return models


def list_models() -> Dict[str, ModelSpec]:
    settings = load_settings()
    return discover_models(settings.models_dir)


def get_model(name: str) -> Optional[ModelSpec]:
    return list_models().get(name)
