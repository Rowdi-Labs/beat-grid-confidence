#!/usr/bin/env bash
# Build pipeline: generate figures → compile PDF
# Usage: ./paper/build.sh [--figures-only] [--pdf-only]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PAPER_DIR="$SCRIPT_DIR"
FIGURES_DIR="$PAPER_DIR/figures"

cd "$PROJECT_DIR"

FIGURES_ONLY=false
PDF_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --figures-only) FIGURES_ONLY=true ;;
        --pdf-only) PDF_ONLY=true ;;
    esac
done

# Step 1: Generate figures
if [ "$PDF_ONLY" = false ]; then
    echo "=== Generating figures ==="
    source .venv/bin/activate

    CKPT="outputs/v4/checkpoints/bgc-epoch=23-val/confidence_brier=0.0255.ckpt"
    if [ ! -f "$CKPT" ]; then
        # Fall back to any available checkpoint
        CKPT=$(find outputs/ -name "*.ckpt" | sort | tail -1)
        echo "Using checkpoint: $CKPT"
    fi

    python scripts/generate_figures.py \
        --hidden-states-dir data/hidden_states \
        --annotations-dir data/beat_this_annotations \
        --checkpoint "$CKPT" \
        --output-dir "$FIGURES_DIR"

    echo ""
fi

if [ "$FIGURES_ONLY" = true ]; then
    echo "Figures generated. Skipping PDF."
    exit 0
fi

# Step 2: Compile PDF
echo "=== Compiling PDF ==="
cd "$PAPER_DIR"
tectonic main.tex

echo ""
echo "=== Done ==="
echo "PDF: $PAPER_DIR/main.pdf"
echo "Figures: $FIGURES_DIR/"
ls -la "$PAPER_DIR/main.pdf"
