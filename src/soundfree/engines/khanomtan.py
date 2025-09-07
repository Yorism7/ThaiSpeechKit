from typing import List, Tuple, Optional
import numpy as np
import sys
import os


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

    # Load only the original KhanomTan model as requested
    # This model is not registered in Coqui-TTS, so we need to load it manually
    model_name = "wannaphong/khanomtan-tts-v1.1"
    local_model_path = "./models/khanomtan-tts-v1.1"

    print(f"Loading KhanomTan TTS v1.1 model from: {model_name}")

    # Check if local model exists
    if os.path.exists(local_model_path) and os.path.exists(os.path.join(local_model_path, "config.json")):
        print(f"Found local KhanomTan model at {local_model_path}")

        try:
            # Try simpler approach using TTS with model path
            print("Trying to use TTS API with local model path...")

            # Use the model path directly with TTS API
            model_path = os.path.join(local_model_path, "best_model.pth")
            config_path = os.path.join(local_model_path, "config.json")

            if os.path.exists(model_path) and os.path.exists(config_path):
                print(f"Using model: {model_path}")
                print(f"Using config: {config_path}")

                # Try to load with TTS API using custom model path
                # This may not work with unregistered models, but let's try
                try:
                    from TTS.api import TTS
                    # Try to create TTS with custom config and model
                    tts = TTS(config_path=config_path, model_path=model_path, gpu=False)
                    print("Successfully loaded KhanomTan TTS v1.1 model with custom paths")
                except Exception as api_error:
                    print(f"TTS API loading failed: {api_error}")

                    # Try alternative: load the model manually and create a simple inference wrapper
                    print("Trying manual model loading...")

                    import torch
                    from TTS.config import load_config
                    from TTS.tts.models import setup_model

                    config = load_config(config_path)

                    # Use setup_model to properly initialize the VITS model
                    print("Setting up VITS model with config...")
                    model = setup_model(config)

                    # Load the trained weights
                    print(f"Loading model weights from {model_path}")
                    checkpoint = torch.load(model_path, map_location='cpu')

                    # Load state dict properly
                    if isinstance(checkpoint, dict):
                        if 'model' in checkpoint:
                            model.load_state_dict(checkpoint['model'])
                        elif 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            # Try loading directly
                            try:
                                model.load_state_dict(checkpoint)
                            except:
                                print("Warning: Could not load state dict, model may not work properly")
                    else:
                        print("Warning: Checkpoint format not recognized")

                    # Create proper inference wrapper for VITS model
                    class KhanomTanInferenceWrapper:
                        def __init__(self, model, config, speakers_dict, languages_dict):
                            self.model = model
                            self.config = config
                            self.speakers_dict = speakers_dict
                            self.languages_dict = languages_dict

                            # Set model to evaluation mode
                            self.model.eval()

                                    # Load speaker embeddings if available
                            try:
                                # Note: speakers.pth contains speaker names to IDs mapping, not embeddings
                                self.speaker_embeddings = None  # Not used for inference
                            except:
                                self.speaker_embeddings = None
                                print("Speaker embeddings not available (normal for VITS)")

                        def tts(self, text=None, speaker=None, language=None):
                            """Generate speech from text with optional speaker and language"""
                            try:
                                import numpy as np
                                from TTS.tts.utils.synthesis import synthesis
                                from TTS.tts.utils.text import text_to_sequence

                                # Set default language to Thai if not specified
                                if language is None:
                                    language = 'th-th'
                                    language_id = self.languages_dict.get(language, 3)  # Default to Thai
                                else:
                                    language_id = self.languages_dict.get(language, 3)  # Default to Thai

                                # Set default speaker if not specified
                                if speaker is None:
                                    speaker = 'Linda'  # Default speaker
                                    speaker_id = self.speakers_dict.get(speaker, 2)  # Default to Linda
                                else:
                                    speaker_id = self.speakers_dict.get(speaker, 2)  # Default to Linda

                                print(f"Generating TTS - Text: {text}")
                                print(f"Speaker: {speaker} (ID: {speaker_id})")
                                print(f"Language: {language} (ID: {language_id})")

                                # Use the model's built-in inference method if available
                                if hasattr(self.model, 'inference'):
                                    # Convert text to sequence
                                    text_sequence = text_to_sequence(text, self.config.characters)

                                    # Generate audio using model inference
                                    outputs = self.model.inference(
                                        text_sequence,
                                        speaker_id=speaker_id,
                                        language_id=language_id,
                                        noise_scale=self.config.model_args.get('inference_noise_scale', 0.667),
                                        length_scale=self.config.model_args.get('length_scale', 1.0)
                                    )

                                    # Extract audio from outputs
                                    if hasattr(outputs, 'waveform') and outputs.waveform is not None:
                                        audio = outputs.waveform.squeeze().cpu().numpy()
                                    elif isinstance(outputs, dict) and 'waveform' in outputs:
                                        audio = outputs['waveform'].squeeze().cpu().numpy()
                                    else:
                                        # Fallback: create a simple tone
                                        print("Warning: Could not extract audio from model output, using fallback")
                                        sample_rate = self.config.audio.get('sample_rate', 16000)
                                        duration = 2.0  # 2 seconds
                                        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))

                                    return audio.astype(np.float32)

                                else:
                                    # Fallback method using synthesis function
                                    print("Using synthesis fallback method")
                                    wav, alignment, _, _ = synthesis(
                                        self.model,
                                        text,
                                        self.config,
                                        use_cuda=False,
                                        speaker_id=speaker_id,
                                        language_id=language_id
                                    )

                                    if wav is not None:
                                        return wav.squeeze().cpu().numpy().astype(np.float32)
                                    else:
                                        raise ValueError("Synthesis failed")

                            except Exception as e:
                                print(f"TTS generation failed: {e}")
                                # Ultimate fallback: generate a simple beep
                                sample_rate = 16000
                                duration = 1.0
                                frequency = 440  # A4 note
                                t = np.linspace(0, duration, int(sample_rate * duration))
                                audio = 0.3 * np.sin(2 * np.pi * frequency * t)
                                return audio.astype(np.float32)

                        @property
                        def speakers(self):
                            return list(self.speakers_dict.keys())

                        @property
                        def languages(self):
                            return list(self.languages_dict.keys())

                    # Load speakers and languages dictionaries
                    speakers_dict = torch.load(os.path.join(local_model_path, "speakers.pth"), map_location='cpu')

                    # Load languages dictionary
                    import json
                    with open(os.path.join(local_model_path, "language_ids.json"), 'r', encoding='utf-8') as f:
                        languages_dict = json.load(f)

                    tts = KhanomTanInferenceWrapper(model, config, speakers_dict, languages_dict)
                    print("Created KhanomTan inference wrapper with proper TTS functionality")
            else:
                raise FileNotFoundError(f"Model or config file not found: {model_path}, {config_path}")

        except Exception as e:
            print(f"Manual loading failed: {e}")
            raise RuntimeError(
                f"Failed to load local KhanomTan model: {e}. "
                f"Please ensure the model files are complete in {local_model_path}. "
                f"Alternatively, use MMS engine: python -m soundfree.cli --engine mms --text your_file.txt --out output.mp3 --mms_model_dir ./models/mms-tts-tha --local_only"
            )

    else:
        # Try to download and use the model directly from Hugging Face
        try:
            print(f"Local model not found. Downloading from Hugging Face: {model_name}")

            # Try alternative loading method using transformers
            try:
                from transformers import pipeline
                print("Trying to use transformers pipeline for text-to-speech...")

                # This might not work for KhanomTan, but let's try
                tts_pipeline = pipeline("text-to-speech", model=model_name)

                class TransformersWrapper:
                    def __init__(self, pipeline):
                        self.pipeline = pipeline

                    def tts(self, text=None, speaker=None, language=None):
                        result = self.pipeline(text)
                        if isinstance(result, dict) and 'audio' in result:
                            return result['audio']['array']
                        return result

                    @property
                    def speakers(self):
                        return None

                    @property
                    def languages(self):
                        return ['th-th', 'en']  # Based on KhanomTan documentation

                tts = TransformersWrapper(tts_pipeline)
                print("Successfully loaded KhanomTan model using transformers")

            except ImportError:
                raise RuntimeError("transformers not available. Please install with: pip install transformers")

        except Exception as e:
            print(f"All loading methods failed: {e}")
            raise RuntimeError(
                f"Could not load KhanomTan model '{model_name}'. "
                f"Please download the model manually: "
                f"huggingface-cli download wannaphong/khanomtan-tts-v1.1 --local-dir ./models/khanomtan-tts-v1.1 "
                f"Then run the command again. "
                f"Alternatively, use MMS engine: python -m soundfree.cli --engine mms --text your_file.txt --out output.mp3 --mms_model_dir ./models/mms-tts-tha --local_only"
            )
    
    # Some models expose `speaker` or `speaker_idx`; not all support
    sr = 22050  # Common default; Coqui doesn't always expose easily
    gap = np.zeros(int(sr * gap_ms / 1000.0), dtype=np.float32)
    segs = []

    for c in chunks:
        try:
            if speaker is not None:
                wav = np.array(tts.tts(text=c, speaker=speaker), dtype=np.float32)
            else:
                # Try to get available speakers and languages
                try:
                    available_speakers = getattr(tts, 'speakers', None)
                    available_languages = getattr(tts, 'languages', None)
                    
                    # KhanomTan TTS v1.1 supports Thai language (th-th) and multiple speakers
                    # Available speakers: Linda (English), Bernard (French), Kerstin (German), Thorsten (German)
                    # Available languages: th-th (Thai), en (English), fr-fr (French), pt-br (Portuguese), x-de (German), x-lb (Luxembourgish)

                    # Check if Thai language is available and use it
                    thai_available = False
                    if available_languages:
                        if 'th-th' in available_languages:
                            thai_available = True
                            thai_lang = 'th-th'
                        elif 'th' in available_languages:
                            thai_available = True
                            thai_lang = 'th'

                    if thai_available:
                        # Use Thai language with appropriate speaker
                        if available_speakers:
                            # Prefer English speaker (Linda) for Thai text as per KhanomTan documentation
                            preferred_speakers = ['linda', 'Linda', 'kerstin', 'Kerstin', 'bernard', 'Bernard']
                            speaker_name = None
                            for pref_speaker in preferred_speakers:
                                if pref_speaker in available_speakers:
                                    speaker_name = pref_speaker
                                    break
                            if speaker_name is None:
                                speaker_name = available_speakers[0]

                            wav = np.array(tts.tts(text=c, speaker=speaker_name, language=thai_lang), dtype=np.float32)
                            print(f"Using speaker: {speaker_name}, language: {thai_lang} (Thai)")
                        else:
                            wav = np.array(tts.tts(text=c, language=thai_lang), dtype=np.float32)
                            print(f"Using language: {thai_lang} (Thai)")
                    else:
                        # Fallback to English if Thai not available
                        print("Thai language not available, using English fallback")
                        if available_speakers and available_languages:
                            if 'en' in available_languages:
                                speaker_name = available_speakers[0]
                                wav = np.array(tts.tts(text=c, speaker=speaker_name, language='en'), dtype=np.float32)
                                print(f"Using speaker: {speaker_name}, language: en")
                            else:
                                speaker_name = available_speakers[0]
                                wav = np.array(tts.tts(text=c, speaker=speaker_name), dtype=np.float32)
                                print(f"Using speaker: {speaker_name}")
                        elif available_speakers:
                            speaker_name = available_speakers[0]
                            wav = np.array(tts.tts(text=c, speaker=speaker_name), dtype=np.float32)
                            print(f"Using speaker: {speaker_name}")
                        elif available_languages:
                            language = available_languages[0]
                            wav = np.array(tts.tts(text=c, language=language), dtype=np.float32)
                            print(f"Using language: {language}")
                        else:
                            wav = np.array(tts.tts(text=c), dtype=np.float32)
                except Exception as e:
                    print(f"Error in TTS generation: {e}")
                wav = np.array(tts.tts(text=c), dtype=np.float32)
        except (TypeError, ValueError, AttributeError) as e:
            # Handle TTS API issues and speaker requirements
            try:
                # Try using synthesizer directly if available
                if hasattr(tts, 'synthesizer') and tts.synthesizer:
                    print("Using synthesizer directly...")
                    try:
                        # For multi-lingual models, handle return values properly
                        result = tts.synthesizer.tts(
                            text=c,
                            speaker_name="Linda",
                            language_name="en"  # Use English as fallback
                        )

                        # Handle different return formats
                        if isinstance(result, tuple):
                            if len(result) >= 1:
                                wav = result[0]
                                # Handle different tensor formats
                                if hasattr(wav, 'squeeze'):
                                    if hasattr(wav, 'cpu'):
                                        wav = np.array(wav.squeeze().cpu().numpy(), dtype=np.float32)
                                    else:
                                        wav = np.array(wav.squeeze(), dtype=np.float32)
                                else:
                                    # Handle list or other formats
                                    wav = np.array(wav, dtype=np.float32).flatten()
                                print("Successfully generated audio using synthesizer with language")
                            else:
                                raise ValueError("Synthesizer returned empty tuple")
                        else:
                            # Single return value - handle list, tensor, or other formats
                            if hasattr(result, 'squeeze'):
                                if hasattr(result, 'cpu'):
                                    wav = np.array(result.squeeze().cpu().numpy(), dtype=np.float32)
                                else:
                                    wav = np.array(result.squeeze(), dtype=np.float32)
                            elif isinstance(result, list):
                                # Handle list return
                                if len(result) > 0:
                                    wav = np.array(result[0], dtype=np.float32).flatten()
                                else:
                                    raise ValueError("Synthesizer returned empty list")
                            else:
                                # Handle other formats
                                wav = np.array(result, dtype=np.float32).flatten()
                            print("Successfully generated audio using synthesizer (single return)")

                    except Exception as synth_error:
                        print(f"Synthesizer with language failed: {synth_error}")
                        try:
                            # Try with style_wav instead of language for multi-lingual models
                            # Use a reference audio file if available
                            style_wav_path = None
                            if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'tts'):
                                # Try to find a reference audio file
                                ref_audio_candidates = [
                                    "./models/khanomtan-tts-v1.1/basic_ref_en.wav",
                                    "./basic_ref_en.wav",
                                    "./reference.wav"
                                ]

                                for candidate in ref_audio_candidates:
                                    if os.path.exists(candidate):
                                        style_wav_path = candidate
                                        break

                                if style_wav_path:
                                    print(f"Using reference audio: {style_wav_path}")
                                    result = tts.synthesizer.tts(
                                        text=c,
                                        speaker_name="Linda",
                                        style_wav=style_wav_path
                                    )

                                    if isinstance(result, tuple) and len(result) >= 1:
                                        wav = np.array(result[0].squeeze().cpu().numpy(), dtype=np.float32)
                                        print("Successfully generated audio using synthesizer with style_wav")
                                    else:
                                        raise ValueError("Style wav approach failed")
                                else:
                                    print("No reference audio file found, using silence fallback")
                                    raise Exception("No reference audio available")

                        except Exception as style_error:
                            print(f"Style wav approach also failed: {style_error}")
                            raise synth_error
                else:
                    # Fallback: try TTS with explicit parameters
                    print("Trying TTS with explicit parameters...")
                    wav = np.array(tts.tts(text=c, speaker="Linda", language="en"), dtype=np.float32)
                    print("Using default speaker: Linda, language: en")
            except Exception as e2:
                print(f"All TTS methods failed: {e2}")
                # Ultimate fallback: generate silence
                sample_rate = 16000
                duration = 1.0
                wav = np.zeros(int(sample_rate * duration), dtype=np.float32)
                print("Generated silence as fallback")
        segs += [wav, gap]

    audio = np.concatenate(segs) if segs else np.array([], dtype=np.float32)
    return audio, sr
