#!/usr/bin/env bash
set -Eeuo pipefail
# Install GAFU and fetch pinned upstream source. Assets go under GAFU_ASSETS.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="${GAFU_ASSETS:-$ROOT/assets}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
command -v conda >/dev/null || { echo "conda is required (Miniconda/Miniforge)" >&2; exit 2; }
mkdir -p "$ASSETS/upstream" "$ASSETS/models"
create() {
  local env="$1" req="$2"
  conda env list | awk '{print $1}' | grep -qx "$env" || conda create -n "$env" "python=$PYTHON_VERSION" -y
  conda run -n "$env" python -m pip install -U pip
  conda run -n "$env" python -m pip install -r "$ROOT/requirements/$req.txt" --extra-index-url "$TORCH_INDEX"
}
clone_pin() {
  local url="$1" dir="$2" rev="$3"
  if [[ ! -d "$dir/.git" ]]; then git clone "$url" "$dir"; fi
  git -C "$dir" fetch --tags --force origin
  git -C "$dir" checkout --detach "$rev"
}
create gafu-bagel bagel
create gafu-lance lance
create gafu-janus janus_pro
# Public upstream source revisions used by the adapters.
clone_pin https://github.com/bytedance/Lance.git "$ASSETS/upstream/lance" main
clone_pin https://github.com/deepseek-ai/Janus.git "$ASSETS/upstream/janus_pro" 1daa72fa409002d40931bd7b36a9280362469ead
clone_pin https://github.com/bytedance-seed/BAGEL.git "$ASSETS/upstream/bagel" main
printf 'Installed environments and upstream source under %s\n' "$ASSETS"
