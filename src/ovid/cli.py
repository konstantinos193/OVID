from pathlib import Path
import typer
import uvicorn

from .config import load_settings
from .pipeline import VideoPipeline
from .registry import list_models, get_model, list_remote_models, pull_model

app = typer.Typer(add_completion=False)


@app.command()
def models() -> None:
    items = list_models()
    if not items:
        typer.echo("No local models found in models/.")
        raise typer.Exit(code=1)
    for model in items.values():
        typer.echo(f"{model.name} ({model.pipeline}) -> {model.path}")


@app.command()
def registry() -> None:
    items = list_remote_models()
    if not items:
        typer.echo("No registry found. Create registry.json or set OVID_REGISTRY.")
        raise typer.Exit(code=1)
    for model in items.values():
        typer.echo(f"{model.name} -> {model.dir} ({len(model.files)} files)")


@app.command()
def pull(name: str, force: bool = False) -> None:
    try:
        target = pull_model(name, force=force)
    except RuntimeError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(f"Pulled {name} into {target}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run("ovid.server:create_app", host=host, port=port, factory=True)


@app.command()
def generate(
    prompt: str,
    model: str | None = None,
    out: Path | None = None,
    frames: int = 16,
    fps: int = 8,
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    guidance: float = 7.5,
    negative: str | None = None,
    seed: int | None = None,
) -> None:
    models = list_models()
    if not models:
        typer.echo("No local models found in models/.")
        raise typer.Exit(code=1)

    model_spec = get_model(model) if model else next(iter(models.values()))
    if not model_spec:
        typer.echo("Model not found.")
        raise typer.Exit(code=1)

    settings = load_settings()
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = out or (settings.outputs_dir / "ovid-output.mp4")

    try:
        pipeline = VideoPipeline(model_spec)
        pipeline.generate(
            prompt=prompt,
            negative_prompt=negative,
            out_path=out_path,
            frames=frames,
            fps=fps,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=seed,
        )
    except RuntimeError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    typer.echo(f"Wrote {out_path}")
