#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SoundFree CLI
Thai Text-to-Speech (local) with MMS / KhanomTan / F5-TTS backends.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import uuid
import numpy as np

from .config import Config, effective_cache_dir
from .text_utils import split_text_th
from .audio import mix_bgm_and_export


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thai TTS (local models)")
    parser.add_argument("--engine", choices=["mms", "khanomtan", "f5"], default="mms")
    parser.add_argument("--text", type=str, required=True, help="Path to UTF-8 .txt file")
    parser.add_argument("--bgm", type=str, default=None, help="Optional BGM file (mp3/wav)")
    parser.add_argument("--out", type=str, default="article_th.mp3", help="Output mp3 file")
    parser.add_argument("--max_chars", type=int, default=None, help="Max chars per chunk")
    parser.add_argument("--gap_ms", type=int, default=None, help="Gap between chunks (ms)")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--artist", type=str, default=None)
    parser.add_argument("--local_only", action="store_true", help="Use local files only for HF")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="HF cache dir")
    parser.add_argument("--mms_model_dir", type=str, default=None, help="Local dir for MMS model")
    parser.add_argument("--khanomtan_model_id", type=str, default=None, help="Model id/dir for Coqui-TTS")
    parser.add_argument("--device", type=str, default=None, help="torch device (e.g. cuda, cpu)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = Config()

    # Override config from CLI
    if args.max_chars is not None:
        cfg.max_chars = int(args.max_chars)
    if args.gap_ms is not None:
        cfg.gap_ms = int(args.gap_ms)
    if args.title is not None:
        cfg.title = args.title
    if args.artist is not None:
        cfg.artist = args.artist
    if args.local_only:
        cfg.local_only = True
    if args.hf_cache_dir is not None:
        cfg.hf_cache_dir = args.hf_cache_dir
    if args.mms_model_dir is not None:
        cfg.mms_model_dir = args.mms_model_dir
    if args.khanomtan_model_id is not None:
        cfg.khanomtan_model_id = args.khanomtan_model_id

    txt_path = Path(args.text)
    if not txt_path.exists():
        print(f"ไม่พบไฟล์ข้อความ: {txt_path}")
        sys.exit(1)

    text = txt_path.read_text(encoding="utf-8")
    chunks = split_text_th(text, max_chars=cfg.max_chars)
    if not chunks:
        print("ไม่พบข้อความหลังตัด กรุณาตรวจไฟล์ .txt")
        sys.exit(1)

    engine = args.engine.lower()
    audio: np.ndarray
    sr: int

    if engine == "mms":
        from .engines.mms import tts_mms

        audio, sr = tts_mms(
            chunks,
            model_id=cfg.mms_model_id,
            gap_ms=cfg.gap_ms,
            local_only=cfg.local_only,
            model_dir=cfg.mms_model_dir,
            hf_cache_dir=effective_cache_dir(cfg),
            device=args.device,
        )
    elif engine == "khanomtan":
        from .engines.khanomtan_simple import tts_khanomtan_simple as tts_khanomtan

        audio, sr = tts_khanomtan(
            chunks,
            model_id=cfg.khanomtan_model_id,
            gap_ms=cfg.gap_ms,
        )
    else:  # f5
        from .engines.f5 import tts_f5

        tmp_wav = f"f5tmp_{uuid.uuid4().hex}.wav"
        audio, sr = tts_f5(chunks, out_wav=tmp_wav)

    if audio.size == 0:
        print("TTS ไม่ได้ผลลัพธ์เสียง")
        sys.exit(1)

    out_mp3 = mix_bgm_and_export(
        audio,
        sr,
        bgm_path=args.bgm,
        out_mp3=args.out,
        title=cfg.title,
        artist=cfg.artist,
        gain_db_voice=cfg.gain_db_voice,
        gain_db_bgm=cfg.gain_db_bgm,
    )

    dur = len(audio) / float(sr)
    print(
        f"บันทึก: {out_mp3} | ระยะเวลา ~{dur/60.0:.1f} นาที | engine={engine}"
    )


if __name__ == "__main__":
    main()

