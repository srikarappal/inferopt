"""The walk's milestone lines, exactly. Vertical Inference's job history parses
them into each job's Flow tab; a change here is a change there, so it should
be made on purpose, with job_flow.py and its tests in the same breath."""

from inferopt import lines


def test_the_lines_a_flow_is_drawn_from():
    assert lines.incumbent(240.44, "stage 1.3") == "incumbent   240.4 goodput  [stage 1.3]"
    assert lines.start("prefix_caching", 1) == f"  start {'prefix_caching':32s} 1 variant"
    assert lines.start("cuda_graph", 3).endswith(" 3 variants")
    assert lines.skip("kv_cache_fp8", "x" * 90) == f"  skip  {'kv_cache_fp8':32s} {'x' * 70}"
    assert lines.verdict(True, "prefix_caching", 1003.4, 3.174, 1) == \
        "   KEEP prefix_caching                     1003.4 goodput  +317.4%  (1 variant)"
    assert lines.verdict(False, "kv_cache_fp8", 230.0, -0.043, 3) == \
        "  revert kv_cache_fp8                        230.0 goodput    -4.3%  (3 variants)"
    assert lines.refused("torch_compile", "2 equivalence") == \
        f"  revert {'torch_compile':32s} no variant passed: 2 equivalence"
    assert lines.stop("budget guard at x") == "  STOP  budget guard at x"
    assert lines.finalist_peak("graph_capture", 1368.04, 64, 16, 16) == \
        "  stage 2.1  graph_capture: peak 1368.0 at L=64, 16 of 16 levels served within the target"
    assert lines.resumed(12) == "  resume    12 measurements already on disk will be replayed, not relaunched"
    assert lines.predicted(1305.84, True) == "predicted   1305.8 goodput  [the walk starts here]"
    assert lines.predicted(700.0, False) == "predicted   700.0 goodput  [stock stays the start]"


def test_a_node_name_longer_than_its_column_still_has_a_space_after_it():
    name = "a_node_with_a_name_longer_than_thirty_two_characters"
    assert f" {name} " in lines.start(name, 2) and f" {name} " in lines.verdict(True, name, 1.0, 0.5, 2)
