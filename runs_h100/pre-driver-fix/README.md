# Pre-driver-fix 1.7B runs

Measured before two defects in the load driver were found and fixed. Kept as
the *before* side of the comparison, not as results.

- **The convoy** (`62a2955`) — `_closed_loop` started all L workers in the same
  instant, so they finished together and re-fired together. The load was a
  convoy of L requests arriving at once every request-duration, not L in steady
  flight. Goodput at a fixed config varied **10.6x** between two identical
  drives.
- **The constant output length** (`5b72afb`) — every request asked for
  `int(mean_output_tokens)` = 259 tokens, discarding the trace's own
  distribution (mean 259.6, sd 167.7, spanning 37–1464). Identical durations are
  what made the convoy permanent.

## What these numbers can and cannot be used for

**Void.** Every TTFT/ITL percentile and everything derived from them — SLO
attainment, "ships within SLO", the 100%-attainment claims. Every ranking with a
margin under ~1.2x, which includes `pb > seqDAG` (1.11x, against a 1.15x
fixed-config spread) and `aiconfigurator < stock` (1.09x). The chosen operating
point L, which flipped between 128 and 256 launch to launch.

**Still good.** Anything not measured through the load driver: math_500 and
MBPP+ accuracy, equivalence divergence, memory footprint, launch failures, the
H100 pinning/legality finding. Margins above ~2x, such as optimized vs stock
(2.3x). And the **configurations themselves** — the search walked real config
space, so these are still good candidates. Re-measure, do not re-discover.

**No `requests.jsonl.gz`.** These predate per-request capture, so their SLO
cannot be replayed — see `docs/slo-replay.md`.

## The replacement

`./rerun_all.sh` writes to `rerun-1.7b-*` alongside this directory. The 14B and
30B-MoE runs still at the top level of `runs/` have the same problem and have
not been moved.

See `docs/decision-log.md` for the full account.
