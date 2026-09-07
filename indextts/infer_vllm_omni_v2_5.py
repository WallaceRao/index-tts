"""IndexTTS-2.5 inference backend via vLLM-Omni (AsyncOmni).

Provides an async ``infer`` API compatible with ``api_server_v2.py``.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import yaml
from vllm import SamplingParams

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

from vllm_omni import AsyncOmni
from vllm_omni.model_executor.models.indextts2.prompt_utils import (
    build_indextts2_prefill_prompt_ids,
)


def resolve_omni_model_root(model_dir: str) -> str:
    """Omni expects a native bundle root that contains ``checkpoints/``."""
    model_dir = os.path.abspath(model_dir)
    if os.path.isfile(os.path.join(model_dir, "gpt.pth")):
        parent = os.path.dirname(model_dir)
        return parent if parent else model_dir
    if os.path.isdir(os.path.join(model_dir, "checkpoints")):
        return model_dir
    # Fallback: treat as bundle root anyway
    return model_dir


def _default_deploy_config() -> str:
    candidates = [
        os.environ.get("INDEXTTS_VLLM_DEPLOY_CONFIG", ""),
        "/home/ubuntu/raoyonghui/indextts_2.5/vllm-omni/vllm_omni/deploy/indextts2_5.yaml",
        "/tmp/vllm-omni/vllm_omni/deploy/indextts2_5.yaml",
        str(Path(__file__).resolve().parents[2] / "vllm_omni" / "deploy" / "indextts2_5.yaml"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "indextts2_5.yaml not found. Set INDEXTTS_VLLM_DEPLOY_CONFIG or install vllm-omni."
    )


def _patched_deploy_config(base_path: str, gpu_memory_utilization: float) -> str:
    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for stage in cfg.get("stages", []):
        stage["gpu_memory_utilization"] = float(gpu_memory_utilization)
    fd, out_path = tempfile.mkstemp(prefix="indextts2_5_deploy_", suffix=".yaml")
    os.close(fd)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def _extract_audio(mm: dict) -> tuple[torch.Tensor | None, int]:
    audio = mm.get("audio")
    if audio is None:
        audio = mm.get("model_outputs")
    if isinstance(audio, list):
        chunks = [
            chunk.reshape(-1)
            for chunk in audio
            if isinstance(chunk, torch.Tensor) and chunk.numel() > 0
        ]
        audio = torch.cat(chunks, dim=0) if chunks else None

    sr_val = mm.get("sr")
    if isinstance(sr_val, list):
        sr_val = sr_val[-1] if sr_val else None
    if hasattr(sr_val, "item"):
        sample_rate = int(sr_val.item())
    else:
        sample_rate = int(sr_val) if sr_val is not None else 22050

    return audio if isinstance(audio, torch.Tensor) else None, sample_rate


def _desired_seconds_to_duration_factor(text: str, desired_seconds: Optional[float]) -> float:
    if desired_seconds is None or desired_seconds <= 0:
        return 1.0
    est = max(0.8, len(text) / 4.5)
    factor = float(desired_seconds) / est
    return max(0.5, min(2.0, factor))


class IndexTTS2:
    """vLLM-Omni backed IndexTTS-2.5 engine used by ``api_server_v2``."""

    def __init__(
        self,
        model_dir: str = "checkpoints",
        is_fp16: bool = False,
        gpu_memory_utilization: float = 0.10,
        temperature: float = 0.3,
        top_p: float = 0.8,
        top_k: int = 30,
        deploy_config: str | None = None,
        stage_init_timeout: int = 600,
    ):
        self.model_root = resolve_omni_model_root(model_dir)
        self.is_fp16 = is_fp16  # kept for CLI compatibility; Omni uses its own dtypes
        self.gpu_memory_utilization = gpu_memory_utilization
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        base_cfg = deploy_config or _default_deploy_config()
        self._deploy_config = _patched_deploy_config(base_cfg, gpu_memory_utilization)
        self._omni = AsyncOmni(
            model=self.model_root,
            deploy_config=self._deploy_config,
            stage_init_timeout=stage_init_timeout,
        )
        self._tokenizer_file = self._get_stage0_tokenizer_file()

        # Match official recipe defaults, with temperature defaulting to 0.3
        # for closer parity with index_tts_vllm-2 production settings.
        self._gpt_sampling = SamplingParams(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            max_tokens=1500,
            repetition_penalty=10.0,
            stop_token_ids=[8193],
            detokenize=False,
        )
        self._s2mel_sampling = SamplingParams(
            temperature=0.0,
            max_tokens=65536,
            detokenize=True,
        )

    def _get_stage0_tokenizer_file(self) -> str | None:
        stage_configs = getattr(self._omni, "stage_configs", None)
        if not stage_configs:
            return None
        engine_args = getattr(stage_configs[0], "engine_args", None)
        if engine_args is None:
            return None
        hf_overrides = getattr(engine_args, "hf_overrides", None)
        if hf_overrides is None and isinstance(engine_args, dict):
            hf_overrides = engine_args.get("hf_overrides")
        if hf_overrides is None:
            return None
        if hasattr(hf_overrides, "get"):
            return hf_overrides.get("tokenizer_file")
        return getattr(hf_overrides, "tokenizer_file", None)

    def _build_request(
        self,
        text: str,
        ref_audio_path: str,
        lang: str,
        text_normalization: bool,
        duration_factor: float,
    ) -> dict:
        additional = {
            "text": [text],
            "voice": [str(ref_audio_path)],
            "lang": [lang],
            "text_normalization": [text_normalization],
            "duration_factor": [float(duration_factor)],
        }
        prompt_kwargs = {
            "model_type": "indextts2_5",
            "lang": lang,
            "text_normalization": text_normalization,
        }
        if self._tokenizer_file is not None:
            prompt_kwargs["tokenizer_file"] = self._tokenizer_file
        return {
            "prompt_token_ids": build_indextts2_prefill_prompt_ids(
                self.model_root,
                text,
                **prompt_kwargs,
            ),
            "additional_information": additional,
        }

    async def infer(
        self,
        spk_audio_prompt: str,
        text: str,
        output_path: str,
        lang: str = "en",
        desired_seconds: float | None = None,
        verbose: bool = False,
        text_normalization: bool = True,
        **_kwargs,
    ) -> str | None:
        lang = (lang or "en").strip().lower()
        duration_factor = _desired_seconds_to_duration_factor(text, desired_seconds)
        if verbose:
            print(
                f"[vLLM-Omni] text={text!r} lang={lang} "
                f"ref={spk_audio_prompt} duration_factor={duration_factor}"
            )

        inputs = self._build_request(
            text=text,
            ref_audio_path=spk_audio_prompt,
            lang=lang,
            text_normalization=text_normalization,
            duration_factor=duration_factor,
        )
        sampling_params_list = [self._gpt_sampling, self._s2mel_sampling]

        audio = None
        sr = 22050
        async for omni_out in self._omni.generate(
            inputs,
            request_id=f"tts-{os.getpid()}-{uuid.uuid4().hex}",
            sampling_params_list=sampling_params_list,
        ):
            mm = getattr(omni_out, "multimodal_output", None) or {}
            if not mm:
                continue
            audio, sr = _extract_audio(mm)

        if audio is None:
            return None

        audio_np = audio.detach().float().cpu().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.reshape(-1)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        sf.write(output_path, audio_np, sr)
        return output_path

    async def infer_list(
        self,
        spk_audio_prompts: list[str],
        texts: list[str],
        output_paths: list[str],
        langs: list[str] | None = None,
        desired_seconds: list[float | None] | None = None,
        verbose: bool = False,
        **_kwargs,
    ) -> list[str | None]:
        langs = langs or ["en"] * len(texts)
        desired_seconds = desired_seconds or [None] * len(texts)
        results = []
        for prompt, text, out, lang, ds in zip(
            spk_audio_prompts, texts, output_paths, langs, desired_seconds
        ):
            results.append(
                await self.infer(
                    spk_audio_prompt=prompt,
                    text=text,
                    output_path=out,
                    lang=lang,
                    desired_seconds=ds,
                    verbose=verbose,
                )
            )
        return results

    def shutdown(self) -> None:
        try:
            if hasattr(self._omni, "shutdown"):
                self._omni.shutdown()
        except Exception:
            pass
        try:
            if self._deploy_config and os.path.isfile(self._deploy_config):
                os.remove(self._deploy_config)
        except Exception:
            pass
