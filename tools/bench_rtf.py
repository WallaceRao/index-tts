#!/usr/bin/env python3
"""Benchmark IndexTTS-2.5 vLLM serving RTF across text lengths and concurrency."""

from __future__ import annotations

import argparse
import base64
import io
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import soundfile as sf

TEXTS = {
    5: "你好世界啊",  # 5
    10: "今天天气真不错呀哈哈",  # 10
    20: "欢迎体验语音合成系统性能测试用例一二三四",  # 20
}


def encode_ref(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


def synth_one(client: httpx.Client, url: str, model: str, text: str, ref_b64: str, lang: str):
    payload = {
        "model": model,
        "input": text,
        "response_format": "wav",
        "speed": 1.0,
        "ref_audio": ref_b64,
        "extra_params": {"lang": lang, "text_normalization": True},
    }
    t0 = time.perf_counter()
    resp = client.post(url, json=payload)
    latency = time.perf_counter() - t0
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    audio, sr = sf.read(io.BytesIO(resp.content))
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    duration = float(len(audio) / sr)
    rtf = latency / duration if duration > 0 else float("inf")
    return {
        "latency_s": latency,
        "audio_s": duration,
        "rtf": rtf,
        "bytes": len(resp.content),
        "sr": sr,
        "chars": len(text),
    }


def run_case(api_base: str, model: str, ref_b64: str, text: str, concurrency: int, repeats: int, lang: str):
    url = f"{api_base.rstrip('/')}/v1/audio/speech"
    n = concurrency * repeats
    results = []
    errors = []

    def worker(_i: int):
        with httpx.Client(timeout=600.0) as client:
            return synth_one(client, url, model, text, ref_b64, lang)

    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(worker, i) for i in range(n)]
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001
                errors.append(str(e))
    wall = time.perf_counter() - wall0

    if not results:
        return {
            "ok": 0,
            "err": len(errors),
            "error_sample": errors[:2],
            "wall_s": wall,
        }

    lat = [r["latency_s"] for r in results]
    aud = [r["audio_s"] for r in results]
    rtf = [r["rtf"] for r in results]
    total_audio = sum(aud)
    return {
        "ok": len(results),
        "err": len(errors),
        "error_sample": errors[:2],
        "wall_s": wall,
        "latency_mean": statistics.mean(lat),
        "latency_p50": statistics.median(lat),
        "audio_mean": statistics.mean(aud),
        "rtf_mean": statistics.mean(rtf),
        "rtf_p50": statistics.median(rtf),
        "rtf_min": min(rtf),
        "rtf_max": max(rtf),
        # System throughput RTF: wall / sum(audio); <1 means faster than realtime aggregate
        "system_rtf": wall / total_audio if total_audio > 0 else float("inf"),
        "qps": len(results) / wall if wall > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8092")
    parser.add_argument("--model", default="/home/ubuntu/raoyonghui/indextts_2.5/index-tts")
    parser.add_argument("--ref-audio", default="examples/voice_01.wav")
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--repeats", type=int, default=3, help="batches per concurrency setting")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--chars", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()

    ref_b64 = encode_ref(args.ref_audio)

    # warmup
    print(f"Warmup x{args.warmup} ...")
    with httpx.Client(timeout=600.0) as client:
        for _ in range(args.warmup):
            synth_one(
                client,
                f"{args.api_base.rstrip('/')}/v1/audio/speech",
                args.model,
                TEXTS[5],
                ref_b64,
                args.lang,
            )
    print("Warmup done.\n")

    header = (
        f"{'chars':>5} {'conc':>4} {'ok/n':>6} {'lat_mean':>9} {'audio_mean':>10} "
        f"{'rtf_mean':>9} {'rtf_p50':>8} {'sys_rtf':>8} {'qps':>7}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for nchar in args.chars:
        text = TEXTS[nchar]
        assert len(text) == nchar, f"text len {len(text)} != {nchar}: {text}"
        for conc in args.concurrency:
            print(f"Running chars={nchar} concurrency={conc} ...", flush=True)
            r = run_case(args.api_base, args.model, ref_b64, text, conc, args.repeats, args.lang)
            n = conc * args.repeats
            if r["ok"] == 0:
                print(f"{nchar:>5} {conc:>4} {'0/'+str(n):>6} FAILED {r.get('error_sample')}")
                continue
            line = (
                f"{nchar:>5} {conc:>4} {str(r['ok'])+'/'+str(n):>6} "
                f"{r['latency_mean']:>9.3f} {r['audio_mean']:>10.3f} "
                f"{r['rtf_mean']:>9.3f} {r['rtf_p50']:>8.3f} {r['system_rtf']:>8.3f} {r['qps']:>7.3f}"
            )
            print(line, flush=True)
            rows.append((nchar, conc, r))

    print("\nNotes:")
    print("  rtf_mean/p50 = per-request latency / audio_duration")
    print("  sys_rtf      = wall_time / sum(audio_duration)  (lower is better; concurrency benefit)")
    print(f"  each cell: concurrency * repeats({args.repeats}) requests")


if __name__ == "__main__":
    main()
