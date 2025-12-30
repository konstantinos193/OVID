from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    home: Path
    models_dir: Path
    outputs_dir: Path


def load_settings() -> Settings:
    home = Path(os.getenv("OVID_HOME", Path.cwd())).resolve()
    models_dir = Path(os.getenv("OVID_MODELS", home / "models")).resolve()
    outputs_dir = Path(os.getenv("OVID_OUTPUTS", home / "outputs")).resolve()
    return Settings(home=home, models_dir=models_dir, outputs_dir=outputs_dir)
