#!/usr/bin/env bash
#
# Make a fresh host able to run the ladders. Idempotent -- safe to re-run.
#
#   ./setup.sh
#
# Does the four things a clone alone does not:
#
#   1. installs evalplus into .evalplus-pkgs/, ISOLATED. It must not go in the
#      serving environment: it drags in openai, anthropic and
#      google-generativeai for its own generation backends, and this project has
#      twice been broken by a dependency installed for a side feature (modelopt
#      pulled setuptools 81 against vLLM's <81).
#   2. materializes data/, which is gitignored because the run outputs and
#      generated corpora are large and machine-specific. MBPP+'s test file is only
#      2.6MB but travels the same way, so a clone has prompts and no tests.
#   3. checks the vLLM/torch environment is actually present.
#   4. reports what this GPU can and cannot run, so an unsupported ladder row is
#      known before it costs a model load.
#
# It does NOT install vLLM or torch. Those are large, CUDA-version specific, and
# usually already provisioned on a GPU host -- guessing at them here would be
# more likely to break a working environment than to fix a broken one.

set -uo pipefail
cd "$(dirname "$0")"

# Prefer an INSTALLED package (pip install -e .). Only fall back to src/ when
# there is none, so a bare checkout works with no install step and an
# editable install is never shadowed by a stale tree.
if ! "${PYTHON:-python}" -c "import inferopt" >/dev/null 2>&1; then
    export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
fi

PY=${PYTHON:-python}
note() { printf '\n=== %s ===\n' "$*"; }

note "environment"
$PY - <<'PY'
import sys
print(f"  python      {sys.version.split()[0]}  ({sys.executable})")
missing = []
for m in ("torch", "vllm", "transformers", "httpx", "pydantic", "datasets"):
    try:
        mod = __import__(m)
        print(f"  {m:12s} {getattr(mod, '__version__', 'present')}")
    except ImportError:
        print(f"  {m:12s} MISSING")
        missing.append(m)
if missing:
    print(f"\n  MISSING: {', '.join(missing)}")
    print(f"  Install them into THIS interpreter -- not another env:")
    print(f"      ./install_vllm.sh          # picks the CUDA build your driver runs")
    print(f"      {sys.executable} -m pip install -r requirements.txt")
    if "torch" in missing or "vllm" in missing:
        print(f"  torch must match this host's CUDA (cu126/cu128 for H100 sm90,")
        print(f"  cu130 for GB10 sm121). Install the matching wheel from")
        print(f"  pytorch.org FIRST if pip's default build is wrong for this box.")
    print(f"\n  Then verify with the SAME interpreter:")
    print(f"      {sys.executable} -c 'import vllm; print(vllm.__version__)'")
    print(f"  Running -m inferopt.run with a different python than the one vLLM lives in")
    print(f"  is the most common failure on a new host.")
PY

note "GPU"
$PY - <<'PY'
try:
    import torch
    if not torch.cuda.is_available():
        print("  no CUDA device visible"); raise SystemExit
    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"  [{i}] {p.name}  sm{cap[0]}{cap[1]}  {p.total_memory/1e9:.0f} GB")
    if n > 1:
        print(f"\n  {n} GPUs visible. The ladders are single-GPU; pin one with")
        print(f"      export CUDA_VISIBLE_DEVICES=0")
        print(f"  otherwise vLLM may claim memory on all of them.")
except Exception as e:
    print(f"  could not query: {type(e).__name__}: {e}")
PY

note "evalplus (isolated, for MBPP+ scoring)"
if [ -d .evalplus-pkgs ] && $PY -c "
import sys; sys.path.insert(0,'.evalplus-pkgs')
import evalplus.evaluate, evalplus.sanitize" 2>/dev/null; then
    echo "  already installed and importable"
else
    echo "  installing into .evalplus-pkgs/ ..."
    $PY -m pip install -q --target .evalplus-pkgs --no-deps \
        evalplus tempdir appdirs multipledispatch wget termcolor fire \
        tree-sitter tree-sitter-python || echo "  pip failed"
    $PY -c "
import sys; sys.path.insert(0,'.evalplus-pkgs')
import evalplus.evaluate, evalplus.sanitize
print('  ok')" || echo "  STILL not importable -- MBPP+ will not score"
fi

note "weight-quantization producer (int4_awq, nvfp4)"
# Not in requirements.txt, and not isolated either. nvidia-modelopt installs
# straight into the serving env because it has no pin conflicts with vLLM, which
# is why it replaced llmcompressor. The one thing it does break is setuptools:
# it pulls 81.0.0 and vLLM requires <81, so the pin is reinstalled last.
#
# Without this step a host silently loses the entire lossy branch. The H100 ran
# a full 14-arm sweep with "int4_awq and nvfp4 will skip" in its log, so the
# technique worth 23x on the other box was never tried there.
if $PY -c "import modelopt.torch.quantization" 2>/dev/null; then
    echo "  already installed"
else
    echo "  installing nvidia-modelopt, accelerate, then pinning setuptools back ..."
    $PY -m pip install -q nvidia-modelopt accelerate && \
    $PY -m pip install -q "setuptools<81.0.0" || echo "  pip failed"
    $PY -c "import modelopt.torch.quantization; print('  ok')" 2>/dev/null || \
        echo "  STILL not importable, int4_awq and nvfp4 will skip"
fi
$PY - <<'CHECK'
import importlib.metadata as md
try:
    v = md.version("setuptools")
    major = int(v.split(".")[0])
    print(f"  setuptools {v}" + ("" if major < 81 else "  <-- BREAKS vLLM, needs <81"))
except Exception:
    pass
CHECK

note "datasets"
if [ -f data/mbpp_plus_full.jsonl ] && [ -f data/math_500.jsonl ]; then
    echo "  data/ already materialized"
    ls -la data/*.jsonl | awk '{printf "    %-34s %6.1f MB\n", $9, $5/1e6}'
else
    echo "  fetching (needs the network once) ..."
    $PY -m inferopt.fetch_data || {
        echo
        echo "  If this failed on SSL certificate verification:"
        echo "      export SSL_CERT_FILE=\$($PY -m certifi)"
        echo "      ./setup.sh"
        echo "  Or copy 2.6MB from a host that has it, no network needed:"
        echo "      rsync <host>:<repo>/data/mbpp_plus_full.jsonl data/"
    }
fi

note "what this GPU can run"
SKIP=$($PY -m inferopt.check_support 2>/dev/null)
if [ -n "$SKIP" ]; then
    echo "  UNSUPPORTED: $SKIP"
    echo "  NVFP4 needs Blackwell FP4 hardware (sm100+). Those ladder rows are"
    echo "  skipped automatically; the rest of the ladder is unaffected."
else
    echo "  all artifact formats supported"
fi

note "self-test"
$PY -m inferopt.selftest 2>&1 | tail -4

cat <<'EOF'

=== ready ===

  export CUDA_VISIBLE_DEVICES=0          # pin ONE GPU on a multi-GPU host

  ./eval_ladder.sh math_500  500 3       # reasoning ladder
  ./eval_ladder.sh mbpp_plus 378 3       # code ladder

  python -m inferopt.summarize runs/ladder-math_500

Artifacts are NOT in the repo -- they are 10-12 GB each. To build the lossy
variants on this host (hours of GPU time):

  python -m inferopt.quantize --model Qwen/Qwen3-14B --trace data/trace_shared.jsonl \
      --produce fp8            # and w4a16, autoquant@6.0, nvfp4 if supported

Without them the ladder measures stock, lossless and fp8, which needs no
artifact -- enough to reproduce the lossless result on new hardware.
EOF
