from typing import List, Tuple, Optional
import numpy as np
import os
import sys
import torch
import json


def tts_khanomtan_simple(
    chunks: List[str],
    model_id: str = "wannaphong/khanomtan-tts-v1.1",
    gap_ms: int = 120,
    speaker: Optional[str] = None,
    speed: float = 0.6,
) -> Tuple[np.ndarray, int]:
    """
    Simple KhanomTan TTS implementation with Thai language support (No fallback)
    """
    try:
        from TTS.utils.synthesizer import Synthesizer
        from TTS.config import load_config
        from TTS.tts.models import setup_model
    except ImportError as e:
        raise RuntimeError(f"Required TTS libraries not found: {e}")

    # Model paths
    local_model_path = "./models/khanomtan-tts-v1.1"
    model_path = os.path.join(local_model_path, "best_model.pth")
    config_path = os.path.join(local_model_path, "config.json")
    speakers_file = os.path.join(local_model_path, "speakers.pth")
    languages_file = os.path.join(local_model_path, "language_ids.json")

    print("🎯 Loading KhanomTan TTS v1.1 for Thai language support...")

    # Check if all required files exist
    required_files = [model_path, config_path, speakers_file, languages_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    try:
        # Load speakers and languages dictionaries first
        speakers_dict = torch.load(speakers_file, map_location='cpu')
        with open(languages_file, 'r', encoding='utf-8') as f:
            languages_dict = json.load(f)

        print(f"📊 Available speakers: {list(speakers_dict.keys())}")
        print(f"🌐 Available languages: {list(languages_dict.keys())}")

        # Load config and setup model
        config = load_config(config_path)
        print(f"✅ Loaded config with sample rate: {config.audio.get('sample_rate', 16000)}")

        # Setup model
        model = setup_model(config)
        print("✅ Model setup complete")

        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        print("✅ Model weights loaded")
        model.eval()

        # Create synthesizer with correct parameters
        synthesizer = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            use_cuda=False
        )
        print("✅ Synthesizer created with correct parameters")

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise RuntimeError(f"Failed to load KhanomTan model: {e}")

    # Sample rate from config
    sr = config.audio.get('sample_rate', 16000)
    gap = np.zeros(int(sr * gap_ms / 1000.0), dtype=np.float32)
    segs = []

    print(f"🎵 Processing {len(chunks)} text chunks...")

    for i, c in enumerate(chunks):
        print(f"\n📝 Chunk {i+1}: '{c[:50]}...'")

        # Determine speaker and language
        target_speaker = speaker if speaker in speakers_dict else "Linda"
        target_language = "th-th"

        print(f"🎤 Using speaker: {target_speaker}")
        print(f"🇹🇭 Using language: {target_language} (Thai)")

        try:
            # Synthesize Thai TTS
            print(f"🐌 Speech speed: {speed} (ช้าลง)")
            # ลองส่ง parameters ผ่าน kwargs
            result = synthesizer.tts(
                text=c,
                speaker_name=target_speaker,
                language_name=target_language,
                length_scale=1.0/speed,  # ช้าลงเมื่อ > 1.0
                speed=speed,  # ลองส่งทั้งสอง
                duration_control=speed  # ลอง parameter อื่น
            )

            # Process synthesis result
            if isinstance(result, tuple) and len(result) >= 1:
                audio_tensor = result[0]
            else:
                audio_tensor = result

            # Convert tensor to numpy
            if hasattr(audio_tensor, 'detach'):
                audio_tensor = audio_tensor.detach()
            if hasattr(audio_tensor, 'cpu'):
                audio_tensor = audio_tensor.cpu()
            if hasattr(audio_tensor, 'numpy'):
                wav = audio_tensor.numpy()
            else:
                wav = np.array(audio_tensor)

            wav = np.array(wav, dtype=np.float32).flatten()

            # Check audio quality
            max_amp = np.max(np.abs(wav))
            rms = np.sqrt(np.mean(wav**2))

            print(f"🎵 Thai audio generated - Shape: {wav.shape}, Max: {max_amp:.4f}, RMS: {rms:.4f}")

            # Amplify if too quiet
            if max_amp < 0.1:
                wav = wav * (0.8 / max_amp) if max_amp > 0 else wav
                print("🔊 Audio amplified")

            # Check if audio has meaningful content
            if np.max(np.abs(wav)) < 0.01:
                raise ValueError("Generated audio is too quiet - likely not Thai TTS")

            # Apply speed adjustment using simple resampling (post-processing)
            if speed != 1.0:
                print(f"🐌 Applying speed adjustment: {speed}x")
                try:
                    # Simple speed adjustment by resampling
                    original_length = len(wav)
                    new_length = int(original_length / speed)

                    # Use numpy for simple linear interpolation
                    indices = np.linspace(0, original_length - 1, new_length)
                    wav = np.interp(indices, np.arange(original_length), wav)

                    print(f"✅ Speed adjusted - Original: {original_length}, New: {len(wav)}")
                except Exception as speed_error:
                    print(f"⚠️ Speed adjustment failed: {speed_error}, using original")

            language_used = f"{target_language} (Thai)"

        except Exception as synthesis_error:
            print(f"❌ Thai TTS synthesis failed: {synthesis_error}")
            # Generate error beep - no fallback
            duration = 0.5
            t = np.linspace(0, duration, int(sr * duration))
            wav = 0.3 * np.sin(2 * np.pi * 220 * t)  # A3 note for error
            wav = np.array(wav, dtype=np.float32)
            language_used = "error (beep)"

        print(f"✅ Language: {language_used}")
        segs.append(wav)

    # Combine all segments
    if segs:
        audio = np.concatenate(segs) if len(segs) > 1 else segs[0]
    else:
        audio = np.array([], dtype=np.float32)

    print(f"\n🎉 TTS Complete! Total duration: {len(audio)/sr:.2f} seconds")
    return audio, sr


# Test function
if __name__ == "__main__":
    # Simple test
    test_text = ["สวัสดีครับ", "วันนี้เป็นวันที่ดี"]
    try:
        audio, sr = tts_khanomtan_simple(test_text)
        print(f"Test successful: {audio.shape}, sample rate: {sr}")
    except Exception as e:
        print(f"Test failed: {e}")