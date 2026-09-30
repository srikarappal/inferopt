"""Every probe render kept, the clip with the still, and the re-render tool's
choice of what to render. See inferopt/diffusion.py and inferopt/render.py."""

import io

import numpy
import pytest
from PIL import Image

from inferopt.diffusion import DiffusionEvaluator, Sample, frames_png, stage_timings
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


def test_the_rerender_takes_each_config_once_with_the_incumbent_then_the_kept_first():
    seed = {"num_inference_steps": 30}
    fewer = {"num_inference_steps": 15}
    cached = {"num_inference_steps": 30, "enable_cache_dit": True}
    result = {"trials": [{"node_id": "cache_dit", "config": cached}, {"node_id": "steps", "config": fewer, "kept": True},
                         {"node_id": "incumbent", "config": seed}, {"node_id": "lossless_complete", "config": seed},
                         {"node_id": "finalist:steps", "config": fewer}]}

    assert measured_configs(result, []) == [("incumbent", seed), ("steps", fewer), ("cache_dit", cached)]
    assert measured_configs(result, ["lossless_complete"]) == [("incumbent", seed)], \
        "the incumbent always renders, since everything is held to its baseline"


def test_a_request_s_stage_timings_come_from_the_server_s_own_timers(tmp_path):
    """Each request's stage lines run together and close on its total; the
    output step is what is left of the least waited requests' totals."""
    log = tmp_path / "server.log"
    lines = []
    for text_s, denoise, decode, total in ((0.9, 11.8, 6.4, 19.5), (0.9, 11.7, 6.5, 44.0), (1.0, 11.9, 6.4, 19.7)):
        lines += [f"[09-29 23:50:01] [TextEncodingStage] finished in {text_s} seconds",
                  f"[09-29 23:50:13] [DenoisingStage] finished in {denoise} seconds",
                  f"[09-29 23:50:19] [DecodingStage] finished in {decode} seconds",
                  f"[09-29 23:50:20] Pixel data generated successfully in \x1b[1m{total}\x1b[0m seconds"]
    log.write_text("\n".join(lines) + "\n")
    stages = stage_timings(log)

    assert stages == {"text": 0.9, "denoise": 11.8, "decode": 6.4, "output": 0.4, "requests": 3}
    assert stage_timings(tmp_path / "absent.log") is None


def test_a_clip_is_scored_on_frames_from_its_first_to_its_last():
    av = pytest.importorskip("av")
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=16)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for shade in range(33):
            frame = av.VideoFrame.from_ndarray(numpy.full((48, 64, 3), shade * 7, dtype=numpy.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    frames = frames_png(buffer.getvalue())

    assert len(frames) == 4, "first, two between and last"
    shades = [Image.open(io.BytesIO(frame)).convert("L").getpixel((32, 24)) for frame in frames]
    assert shades == sorted(shades) and shades[0] < 20 and shades[-1] > 200
