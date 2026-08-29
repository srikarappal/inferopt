#!/bin/bash
#
# Qwen3-0.6B.b


python quantize.py --model Qwen/Qwen3-14B --trace data/trace_shared.jsonl --produce nvfp4
python quantize.py --model Qwen/Qwen3-14B --trace data/trace_shared.jsonl --produce w4a16
python quantize.py --model Qwen/Qwen3-14B --trace data/trace_shared.jsonl --produce autoquant@6.0
python quantize.py --model Qwen/Qwen3-14B --trace data/trace_shared.jsonl --produce autoquant@5.0

for K in nvfp4 w4a16 autoquant_6.0 autoquant_5.0; do
    python eval_repro.py --model artifacts/Qwen__Qwen3-14B--$K \
           --benchmark math_500 --repeats 3 --run-dir runs/quantize/q_$K
done

# FP8 has no artifact -- it's a load-time flag, so it goes through --config
echo '{"gpu_memory_utilization":0.75,"quantization":"fp8"}' > /tmp/fp8.json
python eval_repro.py --model Qwen/Qwen3-14B --config /tmp/fp8.json \
     --benchmark math_500 --repeats 3 --run-dir runs/quantize/q_fp8
