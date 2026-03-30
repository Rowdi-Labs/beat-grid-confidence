"""Extract mel spectrograms from audio files in beat_this format.

Produces .npy files at 22050 Hz, 128 mel bins, hop_length=441 (50 FPS).
These match the format expected by beat_this and our dataset loader.

Usage:
    python scripts/extract_spectrograms.py \
        --audio-dir data/audio/ballroom \
        --output-dir data/spectrograms/ballroom \
        --annotations-dir data/beat_this_annotations/ballroom
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
from rich.console import Console
from rich.progress import track

# beat_this spectrogram parameters
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 441
N_MELS = 128
FMIN = 30.0
FMAX = 11000.0

console = Console()


def extract_spectrogram(audio_path: Path) -> np.ndarray:
    """Extract log-mel spectrogram matching beat_this format.

    Args:
        audio_path: Path to audio file (wav, mp3, au, etc.)

    Returns:
        Spectrogram as [T, 128] float32 array
    """
    # Load and resample to mono 22050 Hz
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=1,  # magnitude, not power
    )

    # Log scale (beat_this uses log10(x * 1000))
    mel = np.log10(mel * 1000 + 1e-10)

    # Transpose to [T, n_mels] for our dataset loader
    mel = mel.T

    return mel.astype(np.float32)


def find_audio_files(audio_dir: Path) -> list[Path]:
    """Find all audio files in a directory (recursive)."""
    extensions = {".wav", ".mp3", ".au", ".flac", ".ogg", ".aif", ".aiff"}
    files = []
    for ext in extensions:
        files.extend(audio_dir.rglob(f"*{ext}"))
    return sorted(files)


def match_annotations(
    audio_files: list[Path],
    annotations_dir: Path,
) -> list[tuple[Path, str]]:
    """Match audio files to annotation stems.

    beat_this annotations use stems like 'ballroom_Albums-AnaBelen_Veneo-01'.
    Audio files may have different naming. We try to match by filename.

    Returns:
        List of (audio_path, annotation_stem) pairs
    """
    # Get all annotation stems
    beats_dir = annotations_dir / "annotations" / "beats"
    if not beats_dir.exists():
        console.print(f"[red]No annotations found at {beats_dir}[/red]")
        return []

    ann_stems = {p.stem for p in beats_dir.glob("*.beats")}

    matched = []
    unmatched_audio = []

    for audio_path in audio_files:
        stem = audio_path.stem

        # Try exact match first
        if stem in ann_stems:
            matched.append((audio_path, stem))
            continue

        # Try with dataset prefix (e.g., "ballroom_" + stem)
        dataset_name = annotations_dir.name
        prefixed = f"{dataset_name}_{stem}"
        if prefixed in ann_stems:
            matched.append((audio_path, prefixed))
            continue

        # Try matching by removing common prefixes/suffixes
        found = False
        for ann_stem in ann_stems:
            if stem in ann_stem or ann_stem in stem:
                matched.append((audio_path, ann_stem))
                found = True
                break

        if not found:
            unmatched_audio.append(audio_path)

    if unmatched_audio:
        console.print(f"[yellow]Warning: {len(unmatched_audio)} audio files could not be matched to annotations[/yellow]")
        for p in unmatched_audio[:5]:
            console.print(f"  {p.name}")
        if len(unmatched_audio) > 5:
            console.print(f"  ... and {len(unmatched_audio) - 5} more")

    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract mel spectrograms from audio")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory containing audio files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for .npy spectrograms")
    parser.add_argument("--annotations-dir", type=Path, default=None, help="Annotations dir (for stem matching)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already have spectrograms")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find audio files
    audio_files = find_audio_files(args.audio_dir)
    console.print(f"Found {len(audio_files)} audio files in {args.audio_dir}")

    if not audio_files:
        console.print("[red]No audio files found.[/red]")
        return

    # Match to annotations if provided
    if args.annotations_dir:
        pairs = match_annotations(audio_files, args.annotations_dir)
        console.print(f"Matched {len(pairs)} files to annotations")
    else:
        pairs = [(f, f.stem) for f in audio_files]

    # Extract spectrograms
    n_extracted = 0
    n_skipped = 0

    for audio_path, stem in track(pairs, description="Extracting spectrograms"):
        output_path = args.output_dir / f"{stem}.npy"

        if args.skip_existing and output_path.exists():
            n_skipped += 1
            continue

        try:
            spec = extract_spectrogram(audio_path)
            np.save(output_path, spec)
            n_extracted += 1
        except Exception as e:
            console.print(f"[red]Error processing {audio_path.name}: {e}[/red]")

    console.print(f"\nDone: {n_extracted} extracted, {n_skipped} skipped")

    # Print stats
    total_size = sum(f.stat().st_size for f in args.output_dir.glob("*.npy"))
    console.print(f"Total spectrogram size: {total_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
