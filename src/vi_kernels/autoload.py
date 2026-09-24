"""Get the adapter into an SGLang server we did not write the entry point of.

`sglang_adapter.install()` monkeypatches modules in the process that calls it,
and an SGLang server is several processes: the entry point spawns a scheduler
(`sgl_diffusion::scheduler`, the LLM scheduler and TP workers) with the spawn
start method, which imports everything afresh. A patch applied in the parent
never reaches the process that runs the kernels.

So the patch is applied at import time instead, in every interpreter of the
environment, and only when asked: a `.pth` file in site-packages imports this
module at interpreter start; this module does nothing unless `VI_KERNELS=1`
is in the environment; when it is, it registers a meta path finder that lets
each target module import normally and applies our wrapper the moment it
has. No torch or sglang import happens until the server itself imports them.

    python -m vi_kernels.autoload --install     # write the .pth once
    VI_KERNELS=1 sglang serve ...               # every process of it picks us up

A launcher makes it a per-launch switch by setting the variable in the
server's environment, which is how a DAG node would turn it on.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import site
import sys
from pathlib import Path

ENV = "VI_KERNELS"
PTH_NAME = "vi_kernels_autoload.pth"
PTH_LINE = f"import os; os.environ.get('{ENV}') == '1' and __import__('vi_kernels.autoload')"

# module that owns the seam -> name of the adapter function that wraps it
TARGETS = {
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe": "_wrap_moe",
    "sglang.multimodal_gen.runtime.layers.layernorm": "_wrap_wan_dit",
    "sglang.multimodal_gen.runtime.models.vaes.wanvae": "_wrap_wan_vae",
    "sglang.multimodal_gen.runtime.models.vaes.wan_vae_cuda_opt": "_wrap_wan_vae",
    "diffusers.models.autoencoders.autoencoder_kl_wan": "_wrap_wan_vae",
}

_callbacks: dict[str, list] = {}
_applied: set[str] = set()


class _AfterImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """A finder that wraps the real loader of a target module so callbacks
    run right after the module's own code has."""

    def find_spec(self, fullname, path, target=None):
        if fullname not in _callbacks or fullname in _applied:
            return None
        # Ask the finders behind us for the real spec, then take over its loader.
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target) if hasattr(finder, "find_spec") else None
            if spec is not None:
                spec._vi_real_loader = spec.loader
                spec.loader = self
                return spec
        return None

    def create_module(self, spec):
        real = spec._vi_real_loader
        return real.create_module(spec) if hasattr(real, "create_module") else None

    def exec_module(self, module):
        real = module.__spec__._vi_real_loader
        real.exec_module(module)
        name = module.__name__
        _applied.add(name)
        for cb in _callbacks.get(name, ()):
            try:
                cb(module)
            except Exception as e:                   # a broken wrap never breaks the server
                print(f"vi_kernels: wrap of {name} failed: {type(e).__name__}: {e}", file=sys.stderr)


_finder = _AfterImport()


def on_import(module_name: str, callback) -> None:
    """Run `callback(module)` once `module_name` is imported; at once if it
    already is."""
    if module_name in sys.modules:
        callback(sys.modules[module_name])
        return
    _callbacks.setdefault(module_name, []).append(callback)
    if _finder not in sys.meta_path:
        sys.meta_path.insert(0, _finder)


def _adapter_callback(fn_name: str):
    def run(module):
        from vi_kernels import sglang_adapter
        result = getattr(sglang_adapter, fn_name)()
        sglang_adapter.installed[fn_name.removeprefix("_wrap_")] = result
        print(f"vi_kernels: {result}  [pid {os.getpid()}]", file=sys.stderr)
    return run


def arm() -> None:
    """Register every seam. Idempotent."""
    for module_name, fn_name in TARGETS.items():
        if module_name not in _callbacks:
            on_import(module_name, _adapter_callback(fn_name))


def pth_path() -> Path:
    return Path(site.getsitepackages()[0]) / PTH_NAME


def install_pth() -> Path:
    path = pth_path()
    path.write_text(PTH_LINE + "\n")
    return path


def main(argv: list[str]) -> int:
    if argv[:1] == ["--install"]:
        print(f"wrote {install_pth()}")
        return 0
    if argv[:1] == ["--uninstall"]:
        path = pth_path()
        if path.exists():
            path.unlink()
        print(f"removed {path}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
elif os.environ.get(ENV) == "1":
    arm()
