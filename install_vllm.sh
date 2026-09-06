#!/usr/bin/env bash
#
# Install vLLM with the CUDA build this host's DRIVER can actually run.
#
#   ./install_vllm.sh
#
# A requirements file cannot do this: it has no way to see the driver, so
# `pip install vllm==0.26.0` always resolves to the PyPI default -- CUDA 13 for
# 0.26. On a 12.x driver that wheel imports and dies on
#
#     ImportError: libcudart.so.13: cannot open shared object file
#
# and it also pulls torch, torchvision, torchaudio and torchcodec as CUDA 13
# builds, so replacing them one at a time fails one import deeper each round:
# torchvision::nms, then _torchaudio.abi3.so, then libtorchcodec_image.so.
#
# CUDA MAJOR versions are the hard boundary. A 12.9 build runs on a 12.6 driver
# through minor-version compatibility; a 13.x build does not run on 12.x at all.
# So this picks by major version and prefers the newest 12.x build available.

set -uo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
VER=${VLLM_VERSION:-0.26.0}
BASE="https://github.com/vllm-project/vllm/releases/download/v${VER}"

drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1)
if [ -z "$cuda" ]; then
    echo "  nvidia-smi reported no CUDA version -- is a driver installed?"
    exit 1
fi
major=${cuda%%.*}
arch=$(uname -m); [ "$arch" = "arm64" ] && arch=aarch64

echo "  driver        $drv  (supports CUDA $cuda)"
echo "  vllm          $VER"

if [ "$major" -ge 13 ]; then
    # The PyPI default is the CUDA 13 build, and torch resolves to a matching
    # cu130 stack on its own. Nothing to override.
    echo "  build         default (cu13x) -- from PyPI"
    $PY -m pip install "vllm==${VER}"
else
    # cu129 is the newest 12.x build vLLM 0.26 publishes. --torch-backend
    # redirects the WHOLE torch family to the matching index; without it pip
    # takes torch from PyPI, which is the cu130 build, and the mismatch returns.
    WHL="${BASE}/vllm-${VER}+cu129-cp38-abi3-manylinux_2_28_${arch}.whl"
    echo "  build         cu129 (newest 12.x; a cu13x wheel cannot run on a 12.x driver)"
    echo "  torch         cu126, via uv --torch-backend"
    command -v uv >/dev/null 2>&1 || $PY -m pip install -q uv
    uv pip install --python "$($PY -c 'import sys;print(sys.executable)')" \
        --torch-backend=cu126 "$WHL" || {
        echo
        echo "  If that 404s, list what this version actually publishes:"
        echo "      gh release view v${VER} --repo vllm-project/vllm --json assets \\"
        echo "         --jq '.assets[].name'"
        exit 1
    }
fi

echo
$PY - <<'PYCHECK'
import torch, vllm
ok = torch.cuda.is_available()
print(f"  vllm {vllm.__version__}   torch {torch.__version__}   "
      f"cuda {torch.version.cuda}   available: {ok}")
if not ok:
    raise SystemExit(
        "\n  torch cannot see a GPU. If it says the driver is too old, the CUDA\n"
        "  MAJOR version of this build is above what the driver supports.")
PYCHECK
