# Voice Cloning TTS

Generate speech audio with a cloned voice using [Fish Speech 1.5](https://github.com/fishaudio/fish-speech).

Give it a few voice samples and a text -- it produces audio in that voice. Works with Russian text.

## Requirements

- **Python** 3.10+
- **GPU:** NVIDIA with 6+ GB VRAM (tested on RTX 5070 12GB)
- **ffmpeg** installed system-wide (for audio conversion)
- **CUDA** 12.8+ (for RTX 50-series) or 12.1+ (for RTX 30/40-series)

## Quick Start

### 1. Set up the environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install PyTorch with CUDA (adjust cu128 to your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install project dependencies
pip install -r requirements.txt
```

> **RTX 5070 note:** If the stable PyTorch build doesn't support your GPU yet, try the nightly:
> ```bash
> pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
> ```

### 2. Add voice samples

Place your voice sample files (mp3, wav, flac, ogg) into the `samples/` directory:

```
samples/
  sample1.mp3
  sample2.mp3
  sample3.mp3
```

### 3. Convert samples

```bash
# Convert all samples to the required WAV format
python convert_samples.py

# Or convert and merge all samples into one file (recommended)
python convert_samples.py --merge
```

This produces mono 44100Hz 16-bit WAV files. The `--merge` flag also creates `samples/combined.wav` from all your samples.

### 4. Generate audio

```bash
# Basic usage
python generate.py --text "Жили-были старик со старухой."

# Specify a particular reference sample
python generate.py --text "Жили-были старик со старухой." --reference samples/sample1.wav

# Read text from a file
python generate.py --text-file story.txt

# Customize output path and generation parameters
python generate.py --text "Текст сказки." \
  --reference samples/combined.wav \
  --output output/fairytale.wav \
  --temperature 0.5 \
  --top-p 0.7
```

### 5. Alternative: Fish Speech WebUI

Fish Speech includes a built-in web interface. This is the easiest way to experiment:

```bash
python -m fish_speech.webui
```

Then open the URL shown in the terminal (usually `http://localhost:7860`), upload your reference audio and type your text.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── convert_samples.py    # mp3/flac/ogg -> wav converter
├── generate.py           # TTS generation script
├── samples/              # Place your voice samples here
│   └── .gitkeep
└── output/               # Generated audio goes here
    └── .gitkeep
```

## Tips for Better Results

- **Clean audio matters.** Remove background noise from samples using [Adobe Podcast Enhance](https://podcast.adobe.com/enhance) or Audacity's Noise Reduction.
- **Try different samples.** One of your recordings may work better than others. Test each one individually.
- **Lower temperature** (0.3-0.5) produces more stable but less expressive speech. Higher values (0.7-0.9) add more variation.
- **Shorter references can work better.** If the full combined sample produces artifacts, try the best 15-20 second clip.
- **Punctuation controls prosody.** Use commas, periods, and ellipses to control pauses and intonation.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `torch.cuda.is_available()` returns False | Reinstall PyTorch with the correct CUDA version |
| RTX 5070 not recognized | Use PyTorch nightly build (see install instructions above) |
| Voice doesn't sound like the reference | Try a different/cleaner sample, lower temperature |
| Audio has artifacts or glitches | Lower temperature to 0.3-0.5, use a cleaner reference |
| Out of memory | Use `--half` flag or reduce reference audio length |
| Wrong pronunciation | Try rewriting the word phonetically |

## Fallback: XTTS v2

If Fish Speech doesn't produce satisfactory results, try [Coqui XTTS v2](https://github.com/coqui-ai/TTS):

```bash
pip install TTS

tts --text "Жили-были старик со старухой." \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav samples/combined.wav \
    --language_idx ru \
    --out_path output/generated_xtts.wav
```
