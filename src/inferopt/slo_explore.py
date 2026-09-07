"""Re-derive the serving metrics at an SLO nobody chose in advance.

The SLO in a run is a guess -- 500ms TTFT and 250ms ITL were picked before
anything was measured. Every number downstream of it (goodput, attainment,
replicas, and therefore cost) is conditional on that guess, and until now
changing the guess meant re-running the GPU.

requests.jsonl.gz carries the per-request record each measurement was collapsed
from, so the whole family of answers is already on disk. This recomputes
summarize() exactly at any threshold:

    python -m inferopt.slo_explore runs/rerun-1.7b-pb --ttft 300 --itl 200
    python -m inferopt.slo_explore runs/rerun-1.7b-pb --sweep --demand-qps 16

--sweep emits the grid an interactive plot needs: one row per (TTFT, ITL,
concurrency), which a slider indexes into rather than recomputing.

The reconstruction is exact, not approximate. There is a test asserting that at
the run's OWN SLO it reproduces the recorded goodput, attainment and p99s to the
last decimal -- if that ever fails, the stored fields are insufficient and this
module is lying rather than degrading.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def load(run_dir: str | Path) -> list[tuple[dict, list[dict]]]:
    """[(meta, rows)] -- one entry per measurement point, in write order."""
    path = Path(run_dir)
    if path.is_dir():
        path = path / "requests.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Runs made before per-request capture landed only "
            f"stored aggregates, and an SLO cannot be moved after the fact from a "
            f"p99 -- goodput counts requests that individually met the bound.")
    out: list[tuple[dict, list[dict]]] = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("_") == "meta":
                out.append((r, []))
            elif out:
                out[-1][1].append(r)
    return out


def _meets(r: dict, ttft_ms: float | None, itl_ms: float | None) -> bool:
    """Req.meets, against the stored fields. Must stay in step with it."""
    if not r["k"] or r["t"] is None:
        return False
    if ttft_ms and r["t"] > ttft_ms:
        return False
    if itl_ms and r["n"] > 1 and (r["l"] - r["t"]) / (r["n"] - 1) > itl_ms:
        return False
    return True


def _pct(xs: list[float], q: float) -> float:
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else float("nan")


def recompute(meta: dict, rows: list[dict], ttft_ms: float | None,
              itl_ms: float | None, demand_qps: float | None = None,
              gpu_hourly_usd: float | None = None) -> dict:
    """summarize() again, at a threshold the run never used."""
    win = max(1e-9, meta["win_s"])
    started = [r for r in rows if 0.0 <= r["s"] < win]
    done = [r for r in started if r["k"] and r["t"] is not None]
    ok = [r for r in rows if _meets(r, ttft_ms, itl_ms)]
    good_tok = sum(r["w"] for r in ok)
    all_tok = sum(r["w"] for r in rows)
    ttfts = sorted(r["t"] for r in done)
    itls = sorted((r["l"] - r["t"]) / (r["n"] - 1) for r in done if r["n"] > 1)
    good_reqs = sum(1 for r in started if _meets(r, ttft_ms, itl_ms))
    out = {
        "node": meta.get("node"), "concurrency": meta.get("L"),
        "phase": meta.get("phase"),
        "ttft_bound_ms": ttft_ms, "itl_bound_ms": itl_ms,
        "goodput": good_tok / win, "throughput": all_tok / win,
        "goodput_req_s": good_reqs / win, "throughput_req_s": len(done) / win,
        # Mirrors summarize(): the denominator is everything that STARTED in
        # the window, so a request that failed counts as a miss. Req.meets()
        # says a failed request does not meet the SLO; the fraction has to
        # agree with the predicate it is a fraction of.
        "slo_attainment": (good_reqs / len(started)) if started else 0.0,
        "ttft_p99_ms": _pct(ttfts, 0.99), "ttft_p95_ms": _pct(ttfts, 0.95),
        "itl_p99_ms": _pct(itls, 0.99), "itl_p95_ms": _pct(itls, 0.95),
        "ttft_n": len(ttfts), "completed": len(done),
    }
    if demand_qps:
        # Little's Law the other way round: how many of THIS operating point
        # does the offered demand need. Ceil, because 2.1 replicas is 3.
        gps = out["goodput_req_s"]
        out["replicas"] = int(-(-demand_qps // gps)) if gps > 0 else None
        if gpu_hourly_usd and out["replicas"]:
            out["usd_per_hour"] = out["replicas"] * gpu_hourly_usd
            tok_h = out["goodput"] * out["replicas"] * 3600
            out["usd_per_mtok"] = (out["usd_per_hour"] / tok_h * 1e6) if tok_h else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("slo_explore")
    ap.add_argument("run_dir")
    ap.add_argument("--ttft", type=float, default=None, help="TTFT bound in ms")
    ap.add_argument("--itl", type=float, default=None, help="ITL bound in ms")
    ap.add_argument("--demand-qps", type=float, default=None,
                    help="offered demand, to turn goodput into a replica count")
    ap.add_argument("--gpu-hourly-usd", type=float, default=None,
                    help="price of ONE GPU-hour; no run captures this, it is yours")
    ap.add_argument("--sweep", action="store_true",
                    help="emit the full (ttft x itl x L) grid as JSON, for a slider")
    ap.add_argument("--ttft-grid", default="100,200,300,500,750,1000,2000")
    ap.add_argument("--itl-grid", default="25,50,100,150,250,500")
    ap.add_argument("--out", default=None, help="write JSON here instead of stdout")
    a = ap.parse_args(argv)

    points = load(a.run_dir)
    if not a.sweep:
        print(f"  {len(points)} measurement points in {a.run_dir}")
        print(f"  {'node':22s} {'L':>5s} {'goodput':>9s} {'req/s':>7s} "
              f"{'attain':>7s} {'ttft p99':>9s} {'repl':>5s}")
        for meta, rows in points:
            m = recompute(meta, rows, a.ttft, a.itl, a.demand_qps, a.gpu_hourly_usd)
            rep = m.get("replicas")
            print(f"  {str(m['node'])[:22]:22s} {m['concurrency']:>5} "
                  f"{m['goodput']:9.1f} {m['goodput_req_s']:7.2f} "
                  f"{m['slo_attainment']:6.1%} {m['ttft_p99_ms']:8.0f}ms "
                  f"{(str(rep) if rep else '-'):>5s}")
        return 0

    grid = []
    for t in [float(x) for x in a.ttft_grid.split(",")]:
        for i in [float(x) for x in a.itl_grid.split(",")]:
            for meta, rows in points:
                grid.append(recompute(meta, rows, t, i, a.demand_qps, a.gpu_hourly_usd))
    blob = json.dumps({"run_dir": str(a.run_dir), "points": len(points), "grid": grid})
    if a.out:
        Path(a.out).write_text(blob)
        print(f"  {len(grid)} rows -> {a.out}")
    else:
        print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
