from typing import List, Tuple, Optional
import numpy as np
import sys


def tts_khanomtan(
    chunks: List[str],
    model_id: str = "wannaphong/khanomtan-tts-v1.1",
    gap_ms: int = 120,
    speaker: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """Synthesize Thai TTS via Coqui-TTS model (KhanomTan).
    Returns: (audio float32 mono, sample_rate)
    """
    try:
        from TTS.api import TTS
    except Exception as e:
        pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            "Coqui-TTS not installed or unsupported on this Python version. "
            f"Detected Python {pyver}. Coqui-TTS currently supports < 3.12. "
            "Create a Python 3.11 virtualenv and install with `pip install TTS`."
        ) from e

    tts = TTS(model_id=model_id, gpu=False)
    # Some models expose `speaker` or `speaker_idx`; not all support
    sr = 22050  # Common default; Coqui doesn't always expose easily
    gap = np.zeros(int(sr * gap_ms / 1000.0), dtype=np.float32)
    segs = []

    for c in chunks:
        try:
            if speaker is not None:
                wav = np.array(tts.tts(text=c, speaker=speaker), dtype=np.float32)
            else:
                wav = np.array(tts.tts(text=c), dtype=np.float32)
        except TypeError:
            # Fallback if signature differs
            wav = np.array(tts.tts(c), dtype=np.float32)
        segs += [wav, gap]

    audio = np.concatenate(segs) if segs else np.array([], dtype=np.float32)
    return audio, sr
