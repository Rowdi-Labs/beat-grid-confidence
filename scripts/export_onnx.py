"""Export trained model (backbone + heads) to ONNX for browser/server deployment."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/beat_grid_confidence.onnx"),
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (match E11even's existing export)",
    )
    args = parser.parse_args()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Opset: {args.opset}")

    # TODO: Implement ONNX export
    # 1. Load model from checkpoint
    # 2. Create dummy input matching mel spectrogram shape [1, T, 128]
    # 3. Export with torch.onnx.export(), including:
    #    - Original beat_this outputs: beat_logits, downbeat_logits
    #    - New head outputs: confidence, tempo_distribution
    # 4. Dynamic axes for batch and time dimensions
    # 5. Validate with onnxruntime
    #
    # Reference: E11even's existing export at scripts/convert-beat-this.sh
    # uses opset 17, dynamic axes, and validates with onnxruntime
    raise NotImplementedError(
        "ONNX export not yet implemented. "
        "Phase 3 milestone: export after heads are trained and validated."
    )


if __name__ == "__main__":
    main()
