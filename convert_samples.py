#!/usr/bin/env python3
"""
Convert audio samples to the format required by Fish Speech TTS.

Converts all mp3/flac/ogg files in the samples/ directory to:
- WAV format
- Mono channel
- 44100 Hz sample rate
- 16-bit PCM

Usage:
    python convert_samples.py
    python convert_samples.py --input-dir my_samples --output-dir converted
    python convert_samples.py --merge  # merge all samples into one file
"""

import argparse
import sys
from pathlib import Path

try:
    from pydub import AudioSegment
except ImportError:
    print("Error: pydub is not installed. Run: pip install pydub")
    print("Also make sure ffmpeg is installed on your system.")
    sys.exit(1)


TARGET_SAMPLE_RATE = 44100
TARGET_CHANNELS = 1  # mono
TARGET_SAMPLE_WIDTH = 2  # 16-bit
SUPPORTED_EXTENSIONS = {".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".wav"}


def convert_file(input_path: Path, output_path: Path) -> bool:
    """Convert a single audio file to the target WAV format."""
    try:
        audio = AudioSegment.from_file(str(input_path))

        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
        audio = audio.set_channels(TARGET_CHANNELS)
        audio = audio.set_sample_width(TARGET_SAMPLE_WIDTH)

        audio.export(str(output_path), format="wav")

        duration_sec = len(audio) / 1000.0
        print(f"  OK: {input_path.name} -> {output_path.name} ({duration_sec:.1f}s)")
        return True
    except Exception as e:
        print(f"  FAIL: {input_path.name} -- {e}")
        return False


def merge_wav_files(wav_files: list[Path], output_path: Path) -> bool:
    """Merge multiple WAV files into a single file with short silence gaps."""
    try:
        silence = AudioSegment.silent(duration=500)  # 0.5s gap between samples
        combined = AudioSegment.empty()

        for i, wav_file in enumerate(sorted(wav_files)):
            audio = AudioSegment.from_wav(str(wav_file))
            if i > 0:
                combined += silence
            combined += audio

        combined.export(str(output_path), format="wav")
        duration_sec = len(combined) / 1000.0
        print(f"\n  Merged {len(wav_files)} files -> {output_path.name} ({duration_sec:.1f}s)")
        return True
    except Exception as e:
        print(f"\n  FAIL merging: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert audio samples for Fish Speech TTS")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("samples"),
        help="Directory with source audio files (default: samples/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for converted files (default: same as input-dir)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Also merge all converted files into combined.wav",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        print(f"Create it and place your audio samples there.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not audio_files:
        print(f"No audio files found in '{input_dir}'.")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio file(s) in '{input_dir}':\n")

    converted_files = []
    for audio_file in sorted(audio_files):
        # Skip files that are already converted WAVs
        output_name = audio_file.stem + ".wav"
        output_path = output_dir / output_name

        if audio_file.suffix.lower() == ".wav" and audio_file == output_path:
            # Re-convert in place to ensure correct format
            pass

        if convert_file(audio_file, output_path):
            converted_files.append(output_path)

    print(f"\nConverted {len(converted_files)}/{len(audio_files)} files.")

    if args.merge and len(converted_files) > 1:
        merge_path = output_dir / "combined.wav"
        merge_wav_files(converted_files, merge_path)
    elif args.merge and len(converted_files) <= 1:
        print("\nSkipping merge: need at least 2 files to merge.")

    print("\nDone! Your samples are ready for Fish Speech TTS.")
    print("Next step: python generate.py --help")


if __name__ == "__main__":
    main()
