# OVID: Local Video Generator (Ollama-style)

OVID is a local-first, model-manager + inference server for text-to-video generation.
Think "Ollama, but for video diffusion": you keep models on disk, run them on your
GPU, and hit a simple HTTP API or UI to generate clips.

## Requirements
- NVIDIA GPU recommended (CUDA-capable).
- Windows/Linux, Python 3.11+ (3.11 recommended).
- Models stored locally on disk.

## Install
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
```

Note: Python 3.14 is not supported yet by key ML packages on Windows.

## Local Models
Place models under `models/` and add a `model.json` for each model folder.

Example folder:
```
models/
  my-local-model/
    model.json
    (model files...)
```

Example `model.json`:
```json
{
  "name": "my-local-model",
  "pipeline": "animatediff",
  "adapter_path": "models/animatediff-adapter",
  "base_model_path": "models/sd15-base"
}
```

## Run (Web UI)
Start the server:
```powershell
.\.venv\Scripts\ovid.exe serve
```
Open:
```
http://127.0.0.1:8000
```

## Run (CLI)
List models:
```powershell
.\.venv\Scripts\ovid.exe models
```

Generate a clip:
```powershell
.\.venv\Scripts\ovid.exe generate --prompt "a neon city at night"
```

## API (Automation)
Base URL: `http://127.0.0.1:8000`

List models:
```
GET /v1/models
```
Response:
```json
{ "models": ["animatediff-local"] }
```

Generate:
```
POST /v1/generate
```
Body:
```json
{
  "prompt": "a neon city at night",
  "model": "animatediff-local",
  "frames": 16,
  "fps": 8,
  "seed": 42
}
```
Response:
```json
{
  "id": "job-id",
  "status": "ok",
  "output": "/outputs/job-id.mp4"
}
```

Fetch output:
```
GET /outputs/{filename}
```

## Notes
- Models are loaded from local disk only.
- The pipeline backend depends on `pipeline` in `model.json`.

## Roadmap
- Model pull/caching with checksums.
- Multiple backends (diffusers, comfy, custom).
- Video upscaling and post-processing.
- Web UI.
