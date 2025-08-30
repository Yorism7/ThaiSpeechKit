from pathlib import Path
from typing import Optional
import numpy as np
from pydub import AudioSegment


def mix_bgm_and_export(
    audio: np.ndarray,
    sr: int,
    bgm_path: Optional[str] = None,
    out_mp3: str = "output.mp3",
    title: str = "Thai TTS Article",
    artist: str = "SoundFree",
    gain_db_voice: float = 0.0,
    gain_db_bgm: float = -18.0,
) -> str:
    # Ensure float32 mono array in [-1, 1]
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.nan_to_num(audio)
    audio = np.clip(audio, -1.0, 1.0)

    # Convert to PCM int16 for pydub raw constructor
    pcm16 = (audio * 32767.0).astype(np.int16)

    voice = AudioSegment(
        pcm16.tobytes(),
        frame_rate=sr,
        sample_width=2,  # int16
        channels=1,
    )
    voice = voice.apply_gain(gain_db_voice)

    if bgm_path and Path(bgm_path).exists():
        bgm = AudioSegment.from_file(bgm_path)
        bgm = bgm.set_frame_rate(sr).set_channels(1).apply_gain(gain_db_bgm)
        loops = int(len(voice) / len(bgm)) + 1 if len(bgm) > 0 else 1
        bgm_long = sum([bgm] * loops) if loops > 1 else bgm
        mixed = bgm_long.overlay(voice)
        final = mixed[: len(voice)]
    else:
        final = voice

    final.export(
        out_mp3,
        format="mp3",
        bitrate="192k",
        tags={"title": title, "artist": artist},
    )
    return out_mp3
