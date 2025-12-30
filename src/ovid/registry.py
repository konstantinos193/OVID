from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional
from urllib.request import urlopen

from .config import load_settings


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    pipeline: str
    extra: Dict[str, str]


@dataclass(frozen=True)
class RemoteFileSpec:
    url: str
    sha256: str
    path: str


@dataclass(frozen=True)
class RemoteModelSpec:
    name: str
    dir: str
    files: list[RemoteFileSpec]


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


def _registry_path() -> Path:
    env_value = os.getenv("OVID_REGISTRY")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path.cwd() / "registry.json"


def _load_registry() -> Dict[str, RemoteModelSpec]:
    reg_path = _registry_path()
    if not reg_path.exists():
        return {}
    with reg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    models = data.get("models", {})
    if not isinstance(models, dict):
        return {}
    out: Dict[str, RemoteModelSpec] = {}
    for name, entry in models.items():
        if not isinstance(entry, dict):
            continue
        dir_name = entry.get("dir", name)
        files = []
        for item in entry.get("files", []):
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            sha256 = item.get("sha256")
            path = item.get("path")
            if not url or not sha256 or not path:
                continue
            files.append(RemoteFileSpec(url=url, sha256=sha256, path=path))
        if files:
            out[name] = RemoteModelSpec(name=name, dir=dir_name, files=files)
    return out


def list_remote_models() -> Dict[str, RemoteModelSpec]:
    return _load_registry()


def get_remote_model(name: str) -> Optional[RemoteModelSpec]:
    return _load_registry().get(name)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_checksum(url: str, dest: Path, sha256: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    with urlopen(url) as resp, tmp_path.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    actual = _sha256_file(tmp_path)
    if actual.lower() != sha256.lower():
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {dest.name}")
    tmp_path.replace(dest)


def pull_model(name: str, force: bool = False) -> Path:
    settings = load_settings()
    registry = list_remote_models()
    spec = registry.get(name)
    if not spec:
        raise RuntimeError(f"Model '{name}' not found in registry.")
    target_dir = settings.models_dir / spec.dir
    for item in spec.files:
        dest = target_dir / item.path
        if dest.exists() and not force:
            if _sha256_file(dest).lower() == item.sha256.lower():
                continue
        _download_with_checksum(item.url, dest, item.sha256)
    return target_dir
