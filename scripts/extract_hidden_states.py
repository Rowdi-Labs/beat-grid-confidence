"""Pre-extract hidden states AND beat/downbeat logits from frozen beat_this backbone.

Saves per-track:
  {stem}.hidden.npy  — [T, 512] float16 hidden states (for head training)
  {stem}.logits.npz  — beat_logits [T], downbeat_logits [T] float32 (for eval + correctness targets)

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
    force: bool = False,
) -> int:
    spec_dir = spectrogram_dir / dataset
    out_dir = output_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(spec_dir.glob("*.npy"))
    overlap = 64

    count = 0
    skipped = 0
    with Progress() as progress:
        task = progress.add_task(f"  {dataset}", total=len(npy_files))

        for npy_path in npy_files:
            stem = npy_path.stem
            hidden_path = out_dir / f"{stem}.hidden.npy"
            logits_path = out_dir / f"{stem}.logits.npz"

            if hidden_path.exists() and logits_path.exists() and not force:
                progress.advance(task)
                count += 1
                continue

            spec = np.load(npy_path, mmap_mode="r")
            spec_tensor = torch.from_numpy(np.array(spec)).float()
            n_frames = spec_tensor.shape[0]

            if chunk_frames <= 0:
                # Full forward pass — no chunking, best quality
                try:
                    with torch.no_grad():
                        inp = spec_tensor.unsqueeze(0).to(device)
                        h = backbone.frontend(inp)
                        h = backbone.transformer_blocks(h)
                        logits = backbone.task_heads(h)
                    hidden = h.squeeze(0).cpu().numpy()
                    beat_logits = logits["beat"].squeeze(0).cpu().numpy()
                    downbeat_logits = logits["downbeat"].squeeze(0).cpu().numpy()
                except RuntimeError as e:
                    if "buffer size" in str(e).lower() or "out of memory" in str(e).lower():
                        console.print(f"  [yellow]OOM {stem} ({n_frames} frames), skipping[/yellow]")
                        skipped += 1
                        progress.advance(task)
                        continue
                    raise
            else:
                # Chunked processing with overlap blending
                hidden_chunks = []
                beat_chunks = []
                downbeat_chunks = []

                start = 0
                while start < n_frames:
                    end = min(start + chunk_frames, n_frames)
                    chunk = spec_tensor[start:end].unsqueeze(0).to(device)

                    with torch.no_grad():
                        h = backbone.frontend(chunk)
                        h = backbone.transformer_blocks(h)
                        logits_out = backbone.task_heads(h)

                    h_np = h.squeeze(0).cpu().numpy()
                    b_np = logits_out["beat"].squeeze(0).cpu().numpy()
                    d_np = logits_out["downbeat"].squeeze(0).cpu().numpy()

                    if hidden_chunks and overlap > 0 and start > 0:
                        blend_len = min(overlap, h_np.shape[0], hidden_chunks[-1].shape[0])
                        if blend_len > 0:
                            w1d = np.linspace(0, 1, blend_len)
                            w2d = w1d.reshape(-1, 1)
                            hidden_chunks[-1][-blend_len:] = (
                                (1 - w2d) * hidden_chunks[-1][-blend_len:] + w2d * h_np[:blend_len]
                            )
                            beat_chunks[-1][-blend_len:] = (
                                (1 - w1d) * beat_chunks[-1][-blend_len:] + w1d * b_np[:blend_len]
                            )
                            downbeat_chunks[-1][-blend_len:] = (
                                (1 - w1d) * downbeat_chunks[-1][-blend_len:] + w1d * d_np[:blend_len]
                            )
                            h_np = h_np[blend_len:]
                            b_np = b_np[blend_len:]
                            d_np = d_np[blend_len:]

                    hidden_chunks.append(h_np)
                    beat_chunks.append(b_np)
                    downbeat_chunks.append(d_np)

                    start = end - overlap if end < n_frames else end

                hidden = np.concatenate(hidden_chunks, axis=0)
                beat_logits = np.concatenate(beat_chunks)
                downbeat_logits = np.concatenate(downbeat_chunks)

            np.save(hidden_path, hidden.astype(np.float16))
            np.savez_compressed(logits_path,
                                beat_logits=beat_logits.astype(np.float32),
                                downbeat_logits=downbeat_logits.astype(np.float32))

            progress.advance(task)
            count += 1

    if skipped:
        console.print(f"  [yellow]Skipped {skipped} tracks (OOM)[/yellow]")

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-extract hidden states + logits from beat_this")
    parser.add_argument("--spectrogram-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default="final0")
    parser.add_argument("--chunk-frames", type=int, default=0,
                        help="Frames per chunk. 0 = full track (no chunking, best quality)")
    parser.add_argument("--force", action="store_true", help="Re-extract even if files exist")
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
            dataset, args.device, args.chunk_frames, args.force,
        )
        total += n
        console.print(f"  Done: {n} tracks")

    console.print(f"\n[bold green]Total: {total} tracks to {args.output_dir}[/bold green]")


if __name__ == "__main__":
    main()
