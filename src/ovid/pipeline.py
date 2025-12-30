from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler

from .registry import ModelSpec


class VideoPipeline:
    def __init__(self, model: ModelSpec) -> None:
        self.model = model

    def generate(
        self,
        prompt: str,
        out_path: Path,
        frames: int = 16,
        fps: int = 8,
        seed: Optional[int] = None,
    ) -> Path:
        if self.model.pipeline != "animatediff":
            raise RuntimeError(
                f"Unsupported pipeline '{self.model.pipeline}'. "
                "Set pipeline to 'animatediff' in model.json."
            )
        return self._generate_animatediff(prompt, out_path, frames, fps, seed)

    def _generate_animatediff(
        self,
        prompt: str,
        out_path: Path,
        frames: int,
        fps: int,
        seed: Optional[int],
    ) -> Path:
        adapter_path = self.model.extra.get("adapter_path")
        base_model_path = self.model.extra.get("base_model_path")
        if not adapter_path or not base_model_path:
            raise RuntimeError(
                "Animatediff requires 'adapter_path' and 'base_model_path' in model.json."
            )

        adapter_dir = Path(adapter_path).resolve()
        base_dir = Path(base_model_path).resolve()
        if not adapter_dir.exists():
            raise RuntimeError(f"Adapter path not found: {adapter_dir}")
        if not base_dir.exists():
            raise RuntimeError(f"Base model path not found: {base_dir}")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for AnimateDiff on Windows.")

        adapter = MotionAdapter.from_pretrained(str(adapter_dir), torch_dtype=torch.float16)
        pipe = AnimateDiffPipeline.from_pretrained(
            str(base_dir), motion_adapter=adapter, torch_dtype=torch.float16
        )
        scheduler = DDIMScheduler.from_pretrained(
            str(base_dir),
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1,
        )
        pipe.scheduler = scheduler
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()

        generator = torch.Generator("cuda")
        if seed is not None:
            generator = generator.manual_seed(seed)

        output = pipe(
            prompt=prompt,
            num_frames=frames,
            guidance_scale=7.5,
            num_inference_steps=20,
            generator=generator,
        )
        vid_frames = output.frames[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arr = [np.array(frame).astype(np.uint8) for frame in vid_frames]
        imageio.mimsave(out_path, arr, fps=fps)
        return out_path
