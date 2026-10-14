#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_VERSION=${PYTHON_VERSION:-3.10}
TORCH_VERSION=${TORCH_VERSION:-2.5.1}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.20.1}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}

command -v conda >/dev/null || { echo "conda is required" >&2; exit 2; }

create_env() {
  local name="$1"
  conda env list | awk '{print $1}' | grep -qx "$name" || conda create -n "$name" "python=$PYTHON_VERSION" -y
  conda run -n "$name" python -m pip install --upgrade pip
  conda run -n "$name" python -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX"
  conda run -n "$name" python -m pip install PyYAML==6.0.2 Pillow==11.3.0 safetensors==0.4.5 pyarrow==11.0.0 numpy==1.24.4 einops==0.8.1
}

install_upstream_requirements() {
  local name="$1"
  local root="$2"
  if [[ -n "$root" && -f "$root/requirements.txt" ]]; then
    conda run -n "$name" python -m pip install -r "$root/requirements.txt"
  fi
  if [[ -n "$root" && -f "$root/pyproject.toml" ]]; then
    conda run -n "$name" python -m pip install -e "$root"
  fi
}

create_env gafu-bagel
create_env gafu-lance
create_env gafu-janus
conda run -n gafu-janus python -m pip install transformers==4.49.0
install_upstream_requirements gafu-bagel "${BAGEL_ROOT:-}"
install_upstream_requirements gafu-lance "${LANCE_ROOT:-}"
install_upstream_requirements gafu-janus "${JANUS_ROOT:-}"

echo "Created gafu-bagel, gafu-lance, and gafu-janus."
