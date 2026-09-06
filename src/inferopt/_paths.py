"""Where the package's data lives, and where the user's work lives.

Two kinds of path, and conflating them is what made this repo un-installable.

  PACKAGE DATA travels with the code and is read-only: dag/llm.json is the
  shipped DAG. It moves with the wheel, so it is reached through
  importlib.resources, never through Path(__file__).parent / "..".

  WORKSPACE DATA belongs to whoever is running: runs/, artifacts/, data/,
  calibration.json, and the isolated dependency trees. It is written, it is
  large -- artifacts are 10-60 GB each -- and it must NOT land inside
  site-packages. It resolves, in order:

      $INFEROPT_HOME               explicit, wins over everything
      ./  when it looks like a workspace   so a git checkout keeps working
                                           exactly as it does today
      platformdirs user data dir           for a pip install

BEFORE THIS, every path was Path(__file__).parent, which means "next to the
source file". That is correct for a checkout and wrong for an installed
package in three ways at once: artifacts would be written into site-packages,
a read-only install would fail on the first run, and two projects sharing one
environment would share one calibration store.
"""

from __future__ import annotations

import os
from pathlib import Path

# A checkout is recognised by things a wheel would not carry: the DAG directory
# alongside a pyproject, or the run outputs of previous work. Deliberately not
# ".git" -- a user may vendor this without the history.
_WORKSPACE_MARKERS = ("pyproject.toml", "dag", "runs")


def _looks_like_workspace(p: Path) -> bool:
    return sum((p / m).exists() for m in _WORKSPACE_MARKERS) >= 2


def home() -> Path:
    """The directory holding this user's runs, artifacts and datasets."""
    env = os.environ.get("INFEROPT_HOME")
    if env:
        return Path(env).expanduser().resolve()

    cwd = Path.cwd().resolve()
    for cand in (cwd, *cwd.parents):
        if _looks_like_workspace(cand):
            return cand

    # Installed, and not run from a checkout.
    try:
        from platformdirs import user_data_dir
        return Path(user_data_dir("inferopt", appauthor=False))
    except Exception:
        return Path(os.environ.get("XDG_DATA_HOME",
                                   Path.home() / ".local" / "share")) / "inferopt"


def workspace(*parts: str, create: bool = False) -> Path:
    """A path under the user's workspace. `create` makes the parent."""
    p = home().joinpath(*parts)
    if create:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def package_file(*parts: str) -> Path:
    """A file that ships WITH the code, e.g. package_file('dag', 'llm.json').

    importlib.resources rather than __file__ arithmetic, so it keeps working
    from a wheel, a zip import, or an editable install.
    """
    try:
        from importlib.resources import files
        r = files("inferopt").joinpath(*parts)
        if r.is_file() or r.is_dir():
            return Path(str(r))
    except Exception:
        pass
    return Path(__file__).resolve().parent.joinpath(*parts)


# Conveniences, so callers do not each spell out the layout and drift.
def data(*parts: str) -> Path:
    return workspace("data", *parts)


def artifacts(*parts: str) -> Path:
    return workspace("artifacts", *parts)


def runs(*parts: str) -> Path:
    return workspace("runs", *parts)


def default_dag() -> Path:
    """The shipped DAG, unless the workspace overrides it with its own."""
    local = workspace("dag", "llm.json")
    return local if local.exists() else package_file("dag", "llm.json")


def describe() -> str:
    src = ("$INFEROPT_HOME" if os.environ.get("INFEROPT_HOME")
           else "checkout" if _looks_like_workspace(home()) else "user data dir")
    return f"{home()}  [{src}]"
