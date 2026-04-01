"""Extract spectrograms + hidden states + logits from audio files.

One-stop pipeline: audio → spectrogram → backbone forward pass → hidden states + logits.
Uses FULL forward pass (no chunking) for artifact-free logits. Batch=1, single-track processing.

Memory per track (~30s, 1500 frames):
    Attention: 1500^2 * 8 heads * 4B * 6 layers = ~432 MB
    Model params: ~81 MB
    Total: ~500 MB peak — fits on M1 16GB

Usage:
    # From a directory of audio files
    python scripts/prepare_audio.py --audio-dir /path/to/audio --dataset-name mydata --output-dir data

    # From mirdata
    python scripts/prepare_audio.py --mirdata gtzan_genre --output-dir data --max-tracks 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress

console = Console()


def extract_spectrogram(audio_path: Path, spect_fn: callable) -> np.ndarray | None:
    """Load audio and compute log-mel spectrogram using beat_this preprocessing."""
    from beat_this.inference import load_audio
    import soxr

    try:
        signal, sr = load_audio(str(audio_path))
    except Exception as e:
        console.print(f"  [red]Failed to load {audio_path.name}: {e}[/red]")
        return None

    if signal.ndim == 2:
        signal = signal.mean(1)
    if sr != 22050:
        signal = soxr.resample(signal, in_rate=sr, out_rate=22050)

    signal_t = torch.tensor(signal, dtype=torch.float32)
    with torch.no_grad():
        spect = spect_fn(signal_t)
    return spect.cpu().numpy()


def full_forward_pass(
    backbone: torch.nn.Module,
    spectrogram: np.ndarray,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Run full-track forward pass through backbone. No chunking = no artifacts.

    Args:
        backbone: beat_this model
        spectrogram: [T, 128] numpy array

    Returns:
        dict with hidden_states [T, 512], beat_logits [T], downbeat_logits [T]
    """
    spec_t = torch.from_numpy(spectrogram).float().unsqueeze(0).to(device)  # [1, T, 128]

    with torch.no_grad():
        h = backbone.frontend(spec_t)
        h = backbone.transformer_blocks(h)
        logits = backbone.task_heads(h)

    return {
        "hidden_states": h.squeeze(0).cpu().numpy(),
        "beat_logits": logits["beat"].squeeze(0).cpu().numpy(),
        "downbeat_logits": logits["downbeat"].squeeze(0).cpu().numpy(),
    }


def process_audio_dir(
    audio_dir: Path,
    dataset_name: str,
    output_dir: Path,
    backbone: torch.nn.Module,
    spect_fn: callable,
    device: str,
    max_tracks: int | None = None,
) -> int:
    """Process all audio files in a directory."""
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"}
    audio_files = sorted(
        f for f in audio_dir.rglob("*") if f.suffix.lower() in extensions
    )

    if max_tracks:
        audio_files = audio_files[:max_tracks]

    spec_dir = output_dir / "spectrograms" / dataset_name
    hidden_dir = output_dir / "hidden_states" / dataset_name
    spec_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with Progress() as progress:
        task = progress.add_task(f"  {dataset_name}", total=len(audio_files))

        for audio_path in audio_files:
            # Normalize stem to match beat_this annotation convention:
            # e.g., "pop.00005" → "pop_00005" (dots to underscores in numbers)
            raw_stem = audio_path.stem.replace(".", "_")
            stem = f"{dataset_name}_{raw_stem}"
            hidden_path = hidden_dir / f"{stem}.hidden.npy"
            logits_path = hidden_dir / f"{stem}.logits.npz"
            spec_path = spec_dir / f"{stem}.npy"

            if hidden_path.exists() and logits_path.exists():
                progress.advance(task)
                count += 1
                continue

            # Extract spectrogram
            spect = extract_spectrogram(audio_path, spect_fn)
            if spect is None:
                progress.advance(task)
                continue

            # Save spectrogram
            np.save(spec_path, spect.astype(np.float32))

            # Full forward pass — no chunking
            try:
                result = full_forward_pass(backbone, spect, device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "buffer size" in str(e).lower():
                    console.print(f"  [yellow]OOM on {stem} ({spect.shape[0]} frames), skipping[/yellow]")
                    progress.advance(task)
                    continue
                raise

            # Save
            np.save(hidden_path, result["hidden_states"].astype(np.float16))
            np.savez_compressed(logits_path,
                                beat_logits=result["beat_logits"].astype(np.float32),
                                downbeat_logits=result["downbeat_logits"].astype(np.float32))

            count += 1
            progress.advance(task)

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare audio: spectrogram + hidden states + logits")
    parser.add_argument("--audio-dir", type=Path, help="Directory of audio files")
    parser.add_argument("--mirdata", type=str, help="mirdata dataset name (e.g., gtzan_genre)")
    parser.add_argument("--dataset-name", type=str, help="Name for the dataset (default: derived from source)")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default="final0")
    parser.add_argument("--max-tracks", type=int, default=None, help="Limit number of tracks")
    args = parser.parse_args()

    if not args.audio_dir and not args.mirdata:
        parser.error("Specify --audio-dir or --mirdata")

    # Load backbone
    console.print(f"[bold]Loading backbone: {args.checkpoint}[/bold]")
    from beat_grid_confidence.model import load_backbone
    backbone, hidden_dim = load_backbone(args.checkpoint, args.device)
    backbone.eval()

    # Get spectrogram function
    from beat_this.preprocessing import LogMelSpect
    spect_fn = LogMelSpect(device=args.device)

    if args.mirdata:
        import mirdata
        dataset_name = args.dataset_name or args.mirdata
        console.print(f"\n[bold]Downloading {args.mirdata} via mirdata...[/bold]")
        mirdata_home = args.output_dir / "mirdata_cache"
        ds = mirdata.initialize(args.mirdata, data_home=str(mirdata_home))
        ds.download()

        # Get audio paths from mirdata
        audio_dir = mirdata_home / args.mirdata
        console.print(f"[bold]Processing {dataset_name}...[/bold]")
        n = process_audio_dir(
            audio_dir, dataset_name, args.output_dir,
            backbone, spect_fn, args.device, args.max_tracks,
        )
        console.print(f"  Done: {n} tracks")

    elif args.audio_dir:
        dataset_name = args.dataset_name or args.audio_dir.name
        console.print(f"\n[bold]Processing {dataset_name} from {args.audio_dir}...[/bold]")
        n = process_audio_dir(
            args.audio_dir, dataset_name, args.output_dir,
            backbone, spect_fn, args.device, args.max_tracks,
        )
        console.print(f"  Done: {n} tracks")

    console.print(f"\n[bold green]Complete. Output in {args.output_dir}/[/bold green]")


if __name__ == "__main__":
    main()
