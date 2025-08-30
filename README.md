SoundFree — Thai Text-to-Speech (Local Models)

Overview
- Local, offline Thai TTS with multiple engines: MMS (Transformers), KhanomTan (Coqui‑TTS), and F5‑TTS CLI.
- Splits long Thai text into chunks, synthesizes audio, optionally mixes background music (BGM), and exports MP3.
- Configuration lives in `src/soundfree/config.py` and can be overridden via CLI flags.

Requirements
- Python: 3.9–3.11 recommended (3.12 is not supported by Coqui‑TTS). Download: https://www.python.org/downloads/
- FFmpeg: Required by `pydub`. Install and ensure it is on your PATH. Download: https://ffmpeg.org/download.html
  - Windows prebuilt binaries: https://www.gyan.dev/ffmpeg/builds/
- Optional (Linux): `libsndfile` system library for `soundfile` if wheels are unavailable.
- PyTorch/Transformers: CPU only by default; for GPU builds install according to PyTorch docs: https://pytorch.org/get-started/locally/

Install (Local Environment)
1) Create and activate a virtual environment.
   - Windows (PowerShell): `python -m venv .venv; . .venv/Scripts/Activate.ps1`
   - macOS/Linux (bash): `python3 -m venv .venv && source .venv/bin/activate`
2) Install core dependencies:
   - `pip install -r requirements.txt`
3) Install the package in editable mode so `python -m soundfree.cli` works from the repo:
   - `pip install -e .`

Optional engines
- KhanomTan (Coqui‑TTS):
  - Requires Python <= 3.11.
  - Install: `pip install TTS`
  - Docs: https://tts.readthedocs.io/
  - Windows may require Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- F5‑TTS CLI:
  - Install per its project instructions and ensure `f5-tts_infer-cli` is on PATH.

Model Download (Offline Use)
- MMS (Transformers): First run will download from Hugging Face. For fully offline use, pre‑download:
  - Install HF CLI: `pip install -U huggingface_hub`
  - Download MMS Thai: `huggingface-cli download facebook/mms-tts-tha --local-dir ./models/mms-tts-tha`
  - Optional KhanomTan model: `huggingface-cli download wannaphong/khanomtan-tts-v1.1 --local-dir ./models/khanomtan-tts-v1.1`
  - Then run with `--mms_model_dir ./models/mms-tts-tha` and `--local_only`.
- KhanomTan (Coqui‑TTS): Will fetch on first use; you may also pass a local directory as the model id.
- F5‑TTS: Managed by its own installer; ensure Thai checkpoints are available and licensed for your use.

Run Examples
- Basic (MMS):
  `python -m soundfree.cli --engine mms --text examples/sample_th.txt --out out_mms.mp3`

- With BGM and metadata:
  `python -m soundfree.cli --engine mms --text examples/sample_th.txt --bgm examples/bgm_sample.mp3 --title "Thai Article" --artist "SoundFree"`

- KhanomTan:
  `python -m soundfree.cli --engine khanomtan --text examples/sample_th.txt --out out_khanomtan.mp3`

- F5‑TTS (CLI installed):
  `python -m soundfree.cli --engine f5 --text examples/sample_th.txt --out out_f5.mp3`

Docker
This repository includes a Docker image for running the CLI without managing local dependencies.

- Build image:
  `docker build -t soundfree:latest .`

- Run MMS (map your working dir for input/output):
  `docker run --rm -v %cd%:/work -w /work soundfree:latest --engine mms --text examples/sample_th.txt --out out_mms.mp3`
  - macOS/Linux: replace `%cd%` with `$(pwd)`.

- Persist HF model cache between runs (recommended):
  `docker run --rm -v $(pwd):/work -v $(pwd)/models:/models -e TRANSFORMERS_CACHE=/models -w /work soundfree:latest --engine mms --text examples/sample_th.txt --out out_mms.mp3`

Project Structure
- `src/soundfree/__init__.py`
- `src/soundfree/config.py`
- `src/soundfree/text_utils.py`
- `src/soundfree/audio.py`
- `src/soundfree/cli.py`
- `src/soundfree/engines/mms.py`
- `src/soundfree/engines/khanomtan.py`
- `src/soundfree/engines/f5.py`
- `examples/sample_th.txt`

Configuration
- Edit defaults in `src/soundfree/config.py` or pass CLI flags:
  - `--engine {mms,khanomtan,f5}`
  - `--text <path>`
  - `--bgm <path>`
  - `--out <path>`
  - `--max_chars <int>`
  - `--gap_ms <int>`
  - `--title <str>`
  - `--artist <str>`
  - `--local_only`
  - `--hf_cache_dir <path>`
  - `--mms_model_dir <path>`
  - `--khanomtan_model_id <id_or_dir>`

Notes
- Pydub requires FFmpeg available on PATH.
- PyTorch/Transformers run on CPU by default; install CUDA builds of torch if you have a supported GPU and driver stack.
- Some terminals may not display Thai logs correctly; audio output is unaffected.

