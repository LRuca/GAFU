#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${GAFU_ASSETS:-$ROOT/assets}"
mkdir -p "$ASSETS/models"
command -v huggingface-cli >/dev/null || { echo "Install huggingface_hub first (run scripts/install.sh)." >&2; exit 2; }
model="${1:-all}"
download() {
  local repo="$1" name="$2"
  huggingface-cli download "$repo" --local-dir "$ASSETS/models/$name" --local-dir-use-symlinks False
}
case "$model" in
  bagel) download ByteDance-Seed/BAGEL-7B-MoT bagel ;;
  lance) download bytedance-research/Lance lance ;;
  janus|janus_pro) download deepseek-ai/Janus-Pro-1B janus_pro ;;
  all) download ByteDance-Seed/BAGEL-7B-MoT bagel; download deepseek-ai/Janus-Pro-1B janus_pro; download bytedance-research/Lance lance ;;
  *) echo "usage: $0 [all|bagel|lance|janus]" >&2; exit 2 ;;
esac
