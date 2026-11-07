#!/usr/bin/env bash
# Download EModelS / EModelX network weights into ./models
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/models"
cd "$ROOT/models"

echo "The checkpoints are already in this repository and are downloaded with Git LFS:"
echo "  models/best_AA_model.ckpt   # EModelS amino-acid model (val_acc=0.512)"
echo "  models/best_BB_model.ckpt   # EModelX backbone model"
echo "  models/best_CA_model.ckpt   # EModelX Cα model"
echo
echo "From the repository root, run: git lfs pull"

# Optional: gdown if the shared archive URL is known
# if command -v gdown >/dev/null 2>&1; then
#   gdown --id 13BKzEBfL0uubYgcJJTGSZ-PkO9hXAB9X -O weights.zip
#   unzip -o weights.zip
# fi
