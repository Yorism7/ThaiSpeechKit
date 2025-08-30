from typing import List, Tuple
import tempfile
import subprocess
import numpy as np
import soundfile as sf


def tts_f5(
    chunks: List[str],
    cli_bin: str = "f5-tts_infer-cli",
    out_wav: str | None = None,
    nfe: int = 32,
    cfg: float = 2.0,
) -> Tuple[np.ndarray, int]:
    """Invoke F5-TTS CLI to synthesize a whole article at once.
    Returns: (audio float32 mono, sample_rate)
    """
    # Write all chunks to a temp text file
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(chunks))
        txt_path = f.name

    tmp_out = out_wav or tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    cmd = [
        cli_bin,
        "--model",
        "F5-TTS",
        "--gen_file",
        txt_path,
        "--out_path",
        tmp_out,
        "--nfe",
        str(nfe),
        "--cfg",
        str(cfg),
    ]
    subprocess.run(cmd, check=True)

    audio, sr = sf.read(tmp_out, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio.astype(np.float32), sr

