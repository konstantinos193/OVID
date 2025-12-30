from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from uuid import uuid4
from pathlib import Path

from .config import load_settings
from .pipeline import VideoPipeline
from .registry import list_models, get_model


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str | None = None
    frames: int = Field(16, ge=1, le=240)
    fps: int = Field(8, ge=1, le=60)
    seed: int | None = None


class GenerateResponse(BaseModel):
    id: str
    status: str
    output: str


def create_app() -> FastAPI:
    app = FastAPI(title="OVID", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>OVID Local UI</title>
  <style>
    :root {
      --bg1: #0b0f14;
      --bg2: #0f172a;
      --bg3: #111827;
      --accent: #7dd3fc;
      --accent-2: #34d399;
      --card: rgba(17,24,39,0.7);
      --text: #e5e7eb;
      --muted: #9ca3af;
      --ring: rgba(125,211,252,0.35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(800px 400px at 15% 15%, rgba(52,211,153,0.12), transparent 60%),
        radial-gradient(900px 500px at 85% 10%, rgba(125,211,252,0.12), transparent 60%),
        linear-gradient(140deg, var(--bg1), var(--bg2), var(--bg3));
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }
    .card {
      width: min(760px, 100%);
      background: var(--card);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 18px;
      padding: 24px;
      backdrop-filter: blur(10px);
      box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    }
    h1 { margin: 0 0 8px; letter-spacing: 0.6px; font-weight: 700; }
    p { margin: 0 0 16px; color: var(--muted); }
    label { display: block; margin: 12px 0 6px; }
    input, select {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(3,7,18,0.6);
      color: var(--text);
      font-size: 16px;
      outline: none;
      transition: border-color 120ms ease, box-shadow 120ms ease;
    }
    input:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--ring);
    }
    .row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }
    button {
      margin-top: 16px;
      padding: 12px 18px;
      border: none;
      border-radius: 12px;
      background: linear-gradient(120deg, var(--accent), var(--accent-2));
      color: #0b1220;
      font-weight: 700;
      cursor: pointer;
      transition: transform 120ms ease, box-shadow 120ms ease;
    }
    button:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(0,0,0,0.35); }
    .status { margin-top: 14px; font-size: 14px; color: var(--muted); }
    .output { margin-top: 10px; }
    .output video { width: 100%; border-radius: 12px; margin-top: 8px; }
    @media (max-width: 640px) {
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>OVID Local UI</h1>
    <p>Generate short clips on your GPU. Models must be on disk.</p>
    <label for="prompt">Prompt</label>
    <input id="prompt" placeholder="a neon city at night" />

    <label for="model">Model</label>
    <select id="model"></select>

    <div class="row">
      <div>
        <label for="frames">Frames</label>
        <input id="frames" type="number" min="1" max="240" value="16" />
      </div>
      <div>
        <label for="fps">FPS</label>
        <input id="fps" type="number" min="1" max="60" value="8" />
      </div>
      <div>
        <label for="seed">Seed (optional)</label>
        <input id="seed" type="number" />
      </div>
    </div>

    <button id="go">Generate</button>
    <div class="status" id="status"></div>
    <div class="output" id="output"></div>
  </div>

  <script>
    const statusEl = document.getElementById("status");
    const outputEl = document.getElementById("output");
    const modelEl = document.getElementById("model");

    async function loadModels() {
      const res = await fetch("/v1/models");
      const data = await res.json();
      modelEl.innerHTML = "";
      (data.models || []).forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        modelEl.appendChild(opt);
      });
    }

    async function generate() {
      statusEl.textContent = "Generating...";
      outputEl.innerHTML = "";
      const payload = {
        prompt: document.getElementById("prompt").value.trim(),
        model: modelEl.value || null,
        frames: parseInt(document.getElementById("frames").value, 10) || 16,
        fps: parseInt(document.getElementById("fps").value, 10) || 8,
        seed: document.getElementById("seed").value
          ? parseInt(document.getElementById("seed").value, 10)
          : null,
      };
      const res = await fetch("/v1/generate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        statusEl.textContent = data.detail || "Failed.";
        return;
      }
      statusEl.textContent = "Done.";
      const video = document.createElement("video");
      video.controls = true;
      video.src = data.output;
      outputEl.appendChild(video);
    }

    document.getElementById("go").addEventListener("click", generate);
    loadModels().catch(() => {
      statusEl.textContent = "Failed to load models.";
    });
  </script>
</body>
</html>
"""

    @app.get("/v1/models")
    def models():
        return {"models": [m.name for m in list_models().values()]}

    @app.get("/outputs/{filename}")
    def outputs(filename: str):
        settings = load_settings()
        target = (settings.outputs_dir / filename).resolve()
        if settings.outputs_dir.resolve() not in target.parents:
            raise HTTPException(status_code=400, detail="Invalid filename.")
        if not target.exists():
            raise HTTPException(status_code=404, detail="Output not found.")
        return FileResponse(target, media_type="video/mp4")

    @app.post("/v1/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest):
        models = list_models()
        if not models:
            raise HTTPException(status_code=400, detail="No local models found in models/.")

        model_spec = get_model(req.model) if req.model else next(iter(models.values()))
        if not model_spec:
            raise HTTPException(status_code=404, detail="Model not found.")

        settings = load_settings()
        settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        job_id = uuid4().hex
        out_path = settings.outputs_dir / f"{job_id}.mp4"

        try:
            pipeline = VideoPipeline(model_spec)
            pipeline.generate(
                prompt=req.prompt,
                out_path=out_path,
                frames=req.frames,
                fps=req.fps,
                seed=req.seed,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

        return GenerateResponse(id=job_id, status="ok", output=f"/outputs/{out_path.name}")

    return app
