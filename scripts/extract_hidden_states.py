"""Pre-extract hidden states from frozen beat_this backbone.

Since the backbone is frozen, hidden states are deterministic per spectrogram.
Extracting them once eliminates the need to run the 20M-param backbone during
training, reducing memory from ~16GB (attention O(T^2)) to ~50MB.

Usage:
    python scripts/extract_hidden_states.py \
        --spectrogram-dir data/spectrograms \
        --output-dir data/hidden_states \
        --datasets ballroom \
        --device cpu \
        --chunk-frames 512
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress

from beat_grid_confidence.model import load_backbone

console = Console()


def extract_for_dataset(
    backbone: torch.nn.Module,
    spectrogram_dir: Path,
    output_dir: Path,
    dataset: str,
    device: str,
    chunk_frames: int = 512,
) -> int:
    """Extract hidden states for all spectrograms in a dataset.

    Processes spectrograms in chunks to avoid OOM on the attention computation.
    Chunks are stitched with overlap to avoid edge artifacts.

    Args:
        backbone: Loaded beat_this model
        spectrogram_dir: Root spectrogram dir (contains dataset subdirs)
        output_dir: Where to save hidden states
        dataset: Dataset name (e.g. "ballroom")
        device: torch device
        chunk_frames: Process this many frames at a time (controls memory)

    Returns:
        Number of tracks processed
    """
    spec_dir = spectrogram_dir / dataset
    out_dir = output_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all spectrograms
    npy_files = sorted(spec_dir.glob("*.npy"))

    overlap = 64  # frames of overlap for stitching

    count = 0
    with Progress() as progress:
        task = progress.add_task(f"  {dataset}", total=len(npy_files))

        for npy_path in npy_files:
            stem = npy_path.stem
            out_path = out_dir / f"{stem}.npy"

            if out_path.exists():
                progress.advance(task)
                count += 1
                continue

            # Load spectrogram [T, 128]
            spec = np.load(npy_path, mmap_mode="r")
            spec_tensor = torch.from_numpy(np.array(spec)).float()
            n_frames = spec_tensor.shape[0]

            # Process in chunks to avoid OOM
            hidden_chunks = []
            start = 0
            while start < n_frames:
                end = min(start + chunk_frames, n_frames)
                chunk = spec_tensor[start:end].unsqueeze(0).to(device)  # [1, T_chunk, 128]

                with torch.no_grad():
                    h = backbone.frontend(chunk)
                    h = backbone.transformer_blocks(h)  # [1, T_chunk, 512]

                h_np = h.squeeze(0).cpu().numpy()

                if hidden_chunks and overlap > 0 and start > 0:
                    # Blend overlapping region
                    blend_len = min(overlap, h_np.shape[0], hidden_chunks[-1].shape[0])
                    if blend_len > 0:
                        weights = np.linspace(0, 1, blend_len).reshape(-1, 1)
                        hidden_chunks[-1][-blend_len:] = (
                            (1 - weights) * hidden_chunks[-1][-blend_len:]
                            + weights * h_np[:blend_len]
                        )
                        h_np = h_np[blend_len:]

                hidden_chunks.append(h_np)

                # Next chunk starts with overlap
                start = end - overlap if end < n_frames else end

            # Concatenate and save
            hidden = np.concatenate(hidden_chunks, axis=0)
            np.save(out_path, hidden.astype(np.float16))  # float16 to save disk

            progress.advance(task)
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-extract hidden states from beat_this backbone")
    parser.add_argument("--spectrogram-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default="final0")
    parser.add_argument("--chunk-frames", type=int, default=512,
                        help="Frames per chunk (lower = less memory, default 512 ~= 10 sec)")
    args = parser.parse_args()

    console.print(f"[bold]Loading backbone: {args.checkpoint}[/bold]")
    backbone, hidden_dim = load_backbone(args.checkpoint, args.device)
    backbone.eval()
    console.print(f"Hidden dim: {hidden_dim}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for dataset in args.datasets:
        console.print(f"\n[bold]Extracting: {dataset}[/bold]")
        n = extract_for_dataset(
            backbone, args.spectrogram_dir, args.output_dir,
            dataset, args.device, args.chunk_frames,
        )
        total += n
        console.print(f"  Done: {n} tracks")

    console.print(f"\n[bold green]Total: {total} tracks extracted to {args.output_dir}[/bold green]")


if __name__ == "__main__":
    main()
