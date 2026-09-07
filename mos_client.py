#!/usr/bin/env python3
"""Simple client for dns_mos_service.py."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any, Optional

import requests

DEFAULT_URL = "http://127.0.0.1:8070/score"


def score_audio(
    audio_path: str,
    url: str = DEFAULT_URL,
    is_personalized_MOS: bool = False,
    timeout: float = 120.0,
) -> Optional[dict[str, Any]]:
    """Call DNSMOS /score API for a local audio path and return the JSON result.

    On any exception, print the traceback and return None.
    """
    try:
        payload = {
            "audio_path": audio_path,
            "is_personalized_MOS": bool(is_personalized_MOS),
        }
        resp = requests.post(url, json=payload, timeout=timeout)
        try:
            data = resp.json()
        except ValueError as e:
            raise RuntimeError(
                f"DNSMOS response is not JSON (status={resp.status_code}): {resp.text}"
            ) from e
        if not resp.ok:
            detail = data.get("detail", data) if isinstance(data, dict) else data
            raise RuntimeError(
                f"DNSMOS request failed (status={resp.status_code}): {detail}"
            )
        return data
    except Exception:
        traceback.print_exc()
        return None




if __name__ == "__main__":
    ret = score_audio("test_input.wav")
    print(ret)
