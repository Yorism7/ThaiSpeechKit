from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Config:
    # General
    default_engine: str = "mms"  # mms | khanomtan | f5
    max_chars: int = 200
    gap_ms: int = 120

    # Output / Mix
    title: str = "Thai TTS Article"
    artist: str = "SoundFree"
    gain_db_voice: float = 0.0
    gain_db_bgm: float = -18.0

    # Hugging Face / local
    local_only: bool = False  # set True to avoid network
    hf_cache_dir: str | None = os.environ.get("HF_HOME") or os.environ.get("HF_CACHE")

    # MMS (Transformers)
    mms_model_id: str = "facebook/mms-tts-tha"
    mms_model_dir: str | None = None  # local path override

    # KhanomTan (Coqui-TTS)
    khanomtan_model_id: str = "wannaphong/khanomtan-tts-v1.1"  # or local dir

    # F5-TTS CLI
    f5_cli_bin: str = "f5-tts_infer-cli"


def effective_cache_dir(cfg: Config) -> str | None:
    if cfg.hf_cache_dir:
        p = Path(cfg.hf_cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    return None

