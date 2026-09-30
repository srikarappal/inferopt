"""Every probe render kept, the clip with the still, and the re-render tool's
choice of what to render. See inferopt/diffusion.py and inferopt/render.py."""

from inferopt.diffusion import DiffusionEvaluator, Sample
from inferopt.render import measured_configs


def an_evaluator(tmp_path):
    """The evaluator's keeping methods need only its run directory."""
    evaluator = DiffusionEvaluator.__new__(DiffusionEvaluator)
    evaluator.run_dir = tmp_path
    evaluator.baseline_dir = tmp_path / "baseline_samples"
    return evaluator


def rendered(index, clip=True):
    sample = Sample(f"prompt {index}", 7 + index)
    sample.image_png = f"png {index}".encode()
    sample.video_mp4 = f"mp4 {index}".encode() if clip else None
    return sample


def test_every_render_is_kept_with_its_clip(tmp_path):
    evaluator = an_evaluator(tmp_path)
    keep = evaluator._keep_renders([rendered(0), rendered(1, clip=False)], "steps-50bc3166")

    assert keep == tmp_path / "launches" / "steps-50bc3166" / "probe"
    assert (keep / "0.png").read_bytes() == b"png 0" and (keep / "0.mp4").read_bytes() == b"mp4 0"
    assert (keep / "1.png").exists() and not (keep / "1.mp4").exists(), "an image has no clip"


def test_the_baseline_keeps_its_clips_too(tmp_path):
    evaluator = an_evaluator(tmp_path)
    evaluator._keep_baseline([rendered(0), rendered(1)])

    assert sorted(path.name for path in evaluator.baseline_dir.iterdir()) == ["0.mp4", "0.png", "1.mp4", "1.png"]


def test_the_rerender_takes_each_config_once_with_the_incumbent_first():
    seed = {"num_inference_steps": 30}
    fewer = {"num_inference_steps": 15}
    result = {"trials": [{"node_id": "steps", "config": fewer}, {"node_id": "incumbent", "config": seed},
                         {"node_id": "lossless_complete", "config": seed}, {"node_id": "finalist:steps", "config": fewer}]}

    assert measured_configs(result, []) == [("incumbent", seed), ("steps", fewer)]
    assert measured_configs(result, ["lossless_complete"]) == [("incumbent", seed)], \
        "the incumbent always renders, since everything is held to its baseline"
