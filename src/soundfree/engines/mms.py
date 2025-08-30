from typing import List, Tuple, Optional
import numpy as np


def tts_mms(
    chunks: List[str],
    model_id: str = "facebook/mms-tts-tha",
    gap_ms: int = 120,
    local_only: bool = False,
    model_dir: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """Synthesize Thai TTS via MMS (Transformers, VITS).

    Returns: (audio float32 mono, sample_rate)
    """
    from transformers import AutoTokenizer, VitsModel
    import torch

    # Load tokenizer/model (prefer local dir if provided)
    if model_dir:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = VitsModel.from_pretrained(model_dir, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, local_files_only=local_only, cache_dir=hf_cache_dir
        )
        model = VitsModel.from_pretrained(
            model_id, local_files_only=local_only, cache_dir=hf_cache_dir
        )

    model.eval()
    if device:
        model = model.to(device)

    sr = model.config.sampling_rate
    gap = np.zeros(int(sr * gap_ms / 1000.0), dtype=np.float32)
    outs = []

    with torch.no_grad():
        for c in chunks:
            inputs = tokenizer(c, return_tensors="pt")
            if device:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            # VITS forward pass returns waveform tensor
            output = model(**inputs)
            audio = (
                output.waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)
            )
            outs += [audio, gap]

    arr = np.concatenate(outs) if outs else np.array([], dtype=np.float32)
    return arr, sr
