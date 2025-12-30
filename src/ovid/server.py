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
    negative_prompt: str | None = None
    model: str | None = None
    frames: int = Field(16, ge=1, le=240)
    fps: int = Field(8, ge=1, le=60)
    width: int = Field(512, ge=128, le=1024)
    height: int = Field(512, ge=128, le=1024)
    steps: int = Field(20, ge=5, le=60)
    guidance: float = Field(7.5, ge=1.0, le=15.0)
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
  <link rel="icon" type="image/svg+xml" href="/favicon.svg"/>
  <style>
    :root {
      --bg1: #0a0d13;
      --bg2: #0f172a;
      --bg3: #111827;
      --accent: #67e8f9;
      --accent-2: #a7f3d0;
      --card: rgba(12,18,30,0.82);
      --text: #e5e7eb;
      --muted: #9ca3af;
      --ring: rgba(103,232,249,0.35);
      --line: rgba(148,163,184,0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(800px 500px at 15% 15%, rgba(103,232,249,0.14), transparent 60%),
        radial-gradient(900px 500px at 85% 10%, rgba(167,243,208,0.12), transparent 60%),
        linear-gradient(140deg, var(--bg1), var(--bg2), var(--bg3));
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .app {
      width: min(1100px, 100%);
      display: grid;
      gap: 16px;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 18px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(12,18,30,0.6);
      backdrop-filter: blur(8px);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 700;
      letter-spacing: 1px;
    }
    .brand-badge {
      width: 34px;
      height: 34px;
      border-radius: 10px;
      display: grid;
      place-items: center;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: #0b1220;
      font-weight: 800;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
    }
    .card {
      background: var(--card);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 18px;
      padding: 24px;
      backdrop-filter: blur(10px);
      box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    }
    .grid {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 16px;
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
    select {
      appearance: none;
      padding-right: 36px;
      background-image:
        linear-gradient(45deg, transparent 50%, #9ca3af 50%),
        linear-gradient(135deg, #9ca3af 50%, transparent 50%),
        linear-gradient(to right, transparent, transparent);
      background-position:
        calc(100% - 18px) 55%,
        calc(100% - 12px) 55%,
        calc(100% - 2.2em) 0.5em;
      background-size: 6px 6px, 6px 6px, 1px 1.5em;
      background-repeat: no-repeat;
    }
    input[type="number"] {
      appearance: textfield;
    }
    input[type="number"]::-webkit-outer-spin-button,
    input[type="number"]::-webkit-inner-spin-button {
      -webkit-appearance: none;
      margin: 0;
    }
    textarea {
      width: 100%;
      min-height: 90px;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(3,7,18,0.6);
      color: var(--text);
      font-size: 16px;
      resize: vertical;
      outline: none;
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
    .row-2 {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
    }
    .row-4 {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
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
    .hint { color: var(--muted); font-size: 12px; margin-top: 6px; }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(15,23,42,0.8);
      border: 1px solid var(--line);
      font-size: 12px;
      color: var(--muted);
    }
    .panel-title {
      margin: 0 0 10px;
      font-size: 14px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .stack { display: grid; gap: 12px; }
    @media (max-width: 640px) {
      .grid { grid-template-columns: 1fr; }
      .row, .row-2, .row-4 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="topbar">
      <div class="brand">
        <div class="brand-badge">O</div>
        <div>OVID</div>
      </div>
      <div class="meta">Local video generation server</div>
    </div>

    <div class="grid">
      <div class="card">
        <h1>Generate Video</h1>
        <p>Local-only workflow. Models are loaded from disk.</p>

        <label for="prompt">Prompt</label>
        <textarea id="prompt" placeholder="A neon city at night, rain-soaked streets, cinematic lighting"></textarea>
        <div class="hint">Tip: Keep prompts short and visual for best results on small GPUs.</div>

        <label for="negative">Negative prompt</label>
        <input id="negative" placeholder="blurry, low quality, distorted" />

        <div class="row-2">
          <div>
            <label for="model">Model</label>
            <select id="model"></select>
          </div>
          <div>
            <label for="preset">Aspect / Resolution</label>
            <select id="preset">
              <option value="512x288">16:9 (512x288)</option>
              <option value="384x384">1:1 (384x384)</option>
              <option value="288x512">9:16 (288x512)</option>
              <option value="512x384">4:3 (512x384)</option>
            </select>
          </div>
        </div>

        <div class="row-4">
          <div>
            <label for="width">Width</label>
            <input id="width" type="number" min="128" max="1024" value="512" />
          </div>
          <div>
            <label for="height">Height</label>
            <input id="height" type="number" min="128" max="1024" value="288" />
          </div>
          <div>
            <label for="fps">FPS</label>
            <input id="fps" type="number" min="1" max="60" value="8" />
          </div>
          <div>
            <label for="duration">Duration (s)</label>
            <input id="duration" type="number" min="1" max="30" value="2" />
          </div>
        </div>

        <div class="row-4">
          <div>
            <label for="frames">Frames</label>
            <input id="frames" type="number" min="1" max="240" value="16" />
          </div>
          <div>
            <label for="steps">Steps</label>
            <input id="steps" type="number" min="5" max="60" value="20" />
          </div>
          <div>
            <label for="guidance">Guidance</label>
            <input id="guidance" type="number" min="1" max="15" step="0.5" value="7.5" />
          </div>
          <div>
            <label for="seed">Seed</label>
            <input id="seed" type="number" />
          </div>
        </div>

        <button id="go">Generate</button>
        <div class="status" id="status"></div>
      </div>

      <div class="card">
        <div class="panel-title">Output</div>
        <div class="chip">MP4 preview</div>
        <div class="output" id="output"></div>
        <div class="panel-title" style="margin-top:16px;">System</div>
        <div class="stack">
          <div class="chip">Local only</div>
          <div class="chip">GPU required</div>
          <div class="chip">Outputs in /outputs</div>
        </div>
        <div class="meta" style="margin-top:16px;">
          Built by <a href="https://github.com/konstantinos193" target="_blank" rel="noopener">konstantinos193</a>
        </div>
      </div>
    </div>
  </div>

  <script>
    const statusEl = document.getElementById("status");
    const outputEl = document.getElementById("output");
    const modelEl = document.getElementById("model");
    const presetEl = document.getElementById("preset");
    const widthEl = document.getElementById("width");
    const heightEl = document.getElementById("height");
    const fpsEl = document.getElementById("fps");
    const durationEl = document.getElementById("duration");
    const framesEl = document.getElementById("frames");

    function applyPreset() {
      const [w, h] = presetEl.value.split("x").map((v) => parseInt(v, 10));
      widthEl.value = w;
      heightEl.value = h;
    }

    function updateFramesFromDuration() {
      const fps = parseInt(fpsEl.value, 10) || 8;
      const duration = parseFloat(durationEl.value);
      if (duration && duration > 0) {
        framesEl.value = Math.max(1, Math.min(240, Math.round(duration * fps)));
      }
    }

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
      updateFramesFromDuration();
      const payload = {
        prompt: document.getElementById("prompt").value.trim(),
        negative_prompt: document.getElementById("negative").value.trim() || null,
        model: modelEl.value || null,
        frames: parseInt(framesEl.value, 10) || 16,
        fps: parseInt(fpsEl.value, 10) || 8,
        width: parseInt(widthEl.value, 10) || 512,
        height: parseInt(heightEl.value, 10) || 512,
        steps: parseInt(document.getElementById("steps").value, 10) || 20,
        guidance: parseFloat(document.getElementById("guidance").value) || 7.5,
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
    presetEl.addEventListener("change", applyPreset);
    fpsEl.addEventListener("change", updateFramesFromDuration);
    durationEl.addEventListener("input", updateFramesFromDuration);
    applyPreset();
    loadModels().catch(() => {
      statusEl.textContent = "Failed to load models.";
    });
  </script>
</body>
</html>
"""

    @app.get("/favicon.svg")
    def favicon():
        icon_path = Path(__file__).with_name("favicon.svg")
        return FileResponse(icon_path, media_type="image/svg+xml")

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
                negative_prompt=req.negative_prompt,
                out_path=out_path,
                frames=req.frames,
                fps=req.fps,
                width=req.width,
                height=req.height,
                steps=req.steps,
                guidance=req.guidance,
                seed=req.seed,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

        return GenerateResponse(id=job_id, status="ok", output=f"/outputs/{out_path.name}")

    return app
