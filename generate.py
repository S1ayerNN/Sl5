#!/usr/bin/env python3
"""
Generate speech audio using Fish Speech TTS with voice cloning.

Uses a reference audio sample to clone the speaker's voice and
synthesize the provided text in that voice.

Usage:
    python generate.py --text "Ваш текст здесь"
    python generate.py --text "Ваш текст здесь" --reference samples/sample1.wav
    python generate.py --text-file story.txt --reference samples/combined.wav
    python generate.py --text "Текст" --reference samples/sample1.wav --output output/result.wav
"""

import argparse
import sys
import time
from pathlib import Path

# 🔥 ИМПОРТ PYTORCH (верните его!)
try:
    import torch
except ImportError:
    print("Error: PyTorch is not installed.")
    print("Install with CUDA 12.8 support:")
    print("  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128")
    sys.exit(1)

# ✅ ВАШ ИСПРАВЛЕННЫЙ БЛОК FISH-SPEECH
try:
    import fish_speech
    
    try:
        from fish_speech.inference import TTSInference
        print("✓ Using TTSInference API")
    except ImportError:
        try:
            from fish_speech.models.vqgan.inference import VQGANInference
            from fish_speech.models.text2semantic.inference import Text2SemanticInference
            print("✓ Using two-stage inference API")
        except ImportError:
            print("✓ Using tools.inference module")
            TTSInference = None
            
except ImportError as e:
    print(f"Error: Cannot import fish_speech module: {e}")
    print("Make sure you installed it with:")
    print("  pip install -e .  (from the fish-speech directory)")
    sys.exit(1)


DEFAULT_MODEL = "fishaudio/fish-speech-1.5"
DEFAULT_REFERENCE = "samples/combined.wav"
DEFAULT_OUTPUT = "output/generated.wav"


def check_gpu():
    """Check GPU availability and print info."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        # 🔧 Исправлено: total_memory вместо total_mem
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        return "cuda"
    else:
        print("WARNING: No GPU detected. Running on CPU (will be slower).")
        return "cpu"


def find_reference_audio(reference_path: str) -> Path:
    """Find a reference audio file, with fallback to samples/ directory."""
    ref = Path(reference_path)
    if ref.exists():
        return ref

    # Try looking in samples/ directory
    samples_dir = Path("samples")
    if samples_dir.exists():
        wav_files = sorted(samples_dir.glob("*.wav"))
        if wav_files:
            print(f"Reference '{reference_path}' not found.")
            print(f"Using first available sample: {wav_files[0]}")
            return wav_files[0]

    print(f"Error: Reference audio '{reference_path}' not found.")
    print("Make sure you have converted your samples first:")
    print("  python convert_samples.py")
    sys.exit(1)


def generate_with_fish_speech(
    text: str,
    reference_audio: Path,
    output_path: Path,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.8,
    repetition_penalty: float = 1.2,
):
    """Generate speech using official fish-speech CLI via subprocess."""
    device = check_gpu()

    print(f"\nModel: {model_name}")
    print(f"Reference audio: {reference_audio}")
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Output: {output_path}")
    print(f"Parameters: temperature={temperature}, top_p={top_p}")
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("Starting generation (first run will download ~3-4 GB)...")
    start_time = time.time()

    # ✅ Запускаем официальный модуль fish-speech через subprocess
    import subprocess
    import sys

    # Путь к папке fish-speech (должна лежать рядом с generate.py)
    fish_speech_dir = Path(__file__).parent / "fish-speech"

    cmd = [
        sys.executable, "-m", "tools.inference",
        "--text", text,
        "--reference", str(reference_audio),
        "--output", str(output_path),
        "--device", device,
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--repetition-penalty", str(repetition_penalty),
    ]

    try:
        # Выполняем команду из директории fish-speech
        result = subprocess.run(cmd, check=True, cwd=str(fish_speech_dir))
        print(f"\n✅ Done in {time.time() - start_time:.1f}s!")
        print(f"Output saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Generation failed. Check logs above.")
        sys.exit(1)

        # Save output
        import soundfile as sf
        sf.write(str(output_path), audio, samplerate=44100)

    except NameError:
        # Fallback: use the two-stage API
        print("Using two-stage inference pipeline...")

        # This is a simplified example. The actual Fish Speech CLI
        # may be the easiest way to run inference. See README for
        # the recommended command-line approach.
        print("\nThe high-level API is not available in this version.")
        print("Use the Fish Speech CLI or WebUI instead:")
        print()
        print("  # Option 1: WebUI (recommended)")
        print("  python -m fish_speech.webui")
        print()
        print("  # Option 2: CLI inference")
        print(f"  python -m fish_speech.inference \\")
        print(f"    --text '{text}' \\")
        print(f"    --reference-audio {reference_audio} \\")
        print(f"    --output {output_path}")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s!")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate speech with Fish Speech TTS + voice cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py --text "Жили-были старик со старухой."
  python generate.py --text "Жили-были старик со старухой." --reference samples/sample1.wav
  python generate.py --text-file story.txt --merge-samples
  python generate.py --text "Hello world" --temperature 0.5 --top-p 0.7
        """,
    )

    # Input text
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", type=str, help="Text to synthesize")
    text_group.add_argument("--text-file", type=Path, help="File containing text to synthesize")

    # Reference audio
    parser.add_argument(
        "--reference",
        type=str,
        default=DEFAULT_REFERENCE,
        help=f"Path to reference audio for voice cloning (default: {DEFAULT_REFERENCE})",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Output audio file path (default: {DEFAULT_OUTPUT})",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Fish Speech model name (default: {DEFAULT_MODEL})",
    )

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.8, help="Top-p sampling (default: 0.8)")
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.2, help="Repetition penalty (default: 1.2)"
    )

    args = parser.parse_args()

    # Read text
    if args.text_file:
        if not args.text_file.exists():
            print(f"Error: Text file '{args.text_file}' not found.")
            sys.exit(1)
        text = args.text_file.read_text(encoding="utf-8").strip()
    else:
        text = args.text

    if not text:
        print("Error: Text is empty.")
        sys.exit(1)

    # Find reference audio
    reference = find_reference_audio(args.reference)

    # Generate
    generate_with_fish_speech(
        text=text,
        reference_audio=reference,
        output_path=args.output,
        model_name=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )


if __name__ == "__main__":
    main()
