"""Render a finished pipeline run's probe set again, one launch per measured
configuration, and keep every render: the still, and the clip for a video.

    python -m inferopt.render runs/diffusion-test-wan2.1-t2v-1.3b-diffusers [--only steps cache_dit]

For runs made before renders were kept for every node (the lossy ones were
scored and dropped) or before a video's mp4 was kept (only its first frame
was). Same prompts, same seeds (7 + i), same served config as the run, so each
render is the one the run scored.

Everything lands under <run>/renders/, a run directory of its own: its own
launches/<tag>/server.log, probe/<i>.png and probe/<i>.mp4, baseline_samples/
and psnr.json against its own baseline. The run's own launches and logs are
left exactly as the search wrote them. Each new baseline still is also held
against the run's original one, which says whether the renders reproduce.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inferopt.diffusion import DiffusionEvaluator, build_context, psnr_db
from inferopt.fingerprint import SLO
from inferopt.request import InferOptRequest, detect_hardware


def measured_configs(result: dict, only: list[str]) -> list[tuple[str, dict]]:
    """(node_id, config) for every distinct config the run measured, the
    incumbent first so the baseline exists before anything is held to it."""
    seen, configs = set(), []
    trials = sorted(result.get("trials") or [], key=lambda trial: trial.get("node_id") != "incumbent")
    for trial in trials:
        key = json.dumps(trial.get("config") or {}, sort_keys=True, default=str)
        if key in seen or (only and trial.get("node_id") not in only and trial.get("node_id") != "incumbent"):
            continue
        seen.add(key)
        configs.append((trial["node_id"], trial["config"]))
    return configs


def reproduces(original: Path, rendered: Path) -> str:
    """The new baseline stills against the run's own, as PSNR per render."""
    values = []
    for still in sorted(rendered.glob("*.png")):
        twin = original / still.name
        if twin.exists():
            values.append(psnr_db(twin.read_bytes(), still.read_bytes()))
    if not values:
        return "no original baseline to hold the new one to"
    shown = ", ".join("identical" if value == float("inf") else f"{value:.1f} dB" for value in values)
    return f"new baseline against the run's own: {shown}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", help="a finished pipeline run: result.json and trace.jsonl")
    parser.add_argument("--only", nargs="*", default=[], help="node ids to render; the incumbent always is")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=8100)
    options = parser.parse_args(argv)

    run_dir = Path(options.run_dir)
    result = json.loads((run_dir / "result.json").read_text())
    trace = run_dir / "trace.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    target_ms = float(result["provenance"]["slo"]["ttft_p99_ms"])
    slo = SLO(ttft_p99_ms=target_ms, itl_p99_ms=None)
    hw = detect_hardware(InferOptRequest(model=result["model"], trace=str(trace), ttft_p99_ms=target_ms))
    fp, _ = build_context(result["model"], rows, slo, 1.0, str(trace), hw)
    out = run_dir / "renders"
    evaluator = DiffusionEvaluator(fp, slo, [row["prompt"] for row in rows], str(out),
                                   gpu=options.gpu, port=options.port, log=print)
    evaluator.profile = False
    for node_id, config in measured_configs(result, options.only):
        tag = evaluator._launch_tag(node_id, config)
        print(f"render      {tag}  {json.dumps(config, default=str)[:120]}", flush=True)
        with evaluator._serve(config, tag):
            samples = evaluator._render_probe_set(config)
        if node_id == "incumbent":
            evaluator._keep_baseline(samples)
            print(f"            {reproduces(run_dir / 'baseline_samples', evaluator.baseline_dir)}", flush=True)
        keep = evaluator._keep_renders(samples, tag)
        divergence = evaluator._equivalence_of(samples, tag)
        failed = [sample.error for sample in samples if sample.error]
        print(f"            {len(list(keep.glob('*.mp4')))} clips, {len(list(keep.glob('*.png')))} stills"
              + (f", {divergence:.0%} differ from the baseline" if divergence is not None else "")
              + (f", {len(failed)} failed: {failed[0]}" if failed else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
