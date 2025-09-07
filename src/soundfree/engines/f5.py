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
        "--output_file",
        tmp_out,
        "--nfe_step",
        str(nfe),
        "--cfg_strength",
        str(cfg),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"F5-TTS output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"F5-TTS error: {e}")
        print(f"F5-TTS stderr: {e.stderr}")
        print(f"F5-TTS stdout: {e.stdout}")
        raise
    
    # Check if the output file was created
    import os
    if not os.path.exists(tmp_out):
        # Try to find the generated file in the current directory
        import glob
        wav_files = glob.glob("f5tmp_*.wav")
        if wav_files:
            tmp_out = wav_files[0]
            print(f"Found generated file: {tmp_out}")
        else:
            raise FileNotFoundError(f"F5-TTS output file not found: {tmp_out}")

    try:
        audio, sr = sf.read(tmp_out, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio.astype(np.float32), sr
    except Exception as e:
        print(f"Error reading audio file {tmp_out}: {e}")
        # Try to read with different method
        try:
            import librosa
            audio, sr = librosa.load(tmp_out, sr=None, mono=True)
            return audio.astype(np.float32), sr
        except ImportError:
            raise RuntimeError(f"Could not read audio file {tmp_out}: {e}")

