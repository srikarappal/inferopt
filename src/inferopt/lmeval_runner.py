"""Run lm-evaluation-harness against a serving endpoint. Executed by ANOTHER interpreter.

    <lmeval-python> src/inferopt/lmeval_runner.py config.json results.json

Invoked BY PATH, never with -m: the module form would import the inferopt
package, whose __init__ pulls pydantic, and this interpreter has no reason
to carry inferopt's dependencies at all.

This file must not import inferopt. It runs inside the environment lm_eval is
installed in, which is deliberately not the environment vLLM is installed in:
lm_eval pulls transformers, datasets, sympy, nltk and antlr, and none of that
belongs beside a serving stack whose dependency graph is already load bearing.
The two talk over HTTP and a pair of JSON files, so neither constrains the
other's versions.

Config keys, all of which come from inferopt.leaderboard:

    tasks          list of lm_eval task or group ids
    model          model name to send in the request body
    base_url       .../v1/completions of an ALREADY-RUNNING server
    limit          cap the number of documents per task, or null for all
    num_fewshot    override the task's own shot count, or null to keep it
    gen_kwargs     e.g. {"max_gen_toks": 8192}, or null to keep the task's own
    max_length     prompt+generation ceiling, must fit the served max_model_len
    concurrent     in-flight requests
"""
import json
import sys


def main() -> int:
    cfg = json.loads(open(sys.argv[1]).read())
    out_path = sys.argv[2]

    import lm_eval

    model_args = {
        "model": cfg["model"],
        "base_url": cfg["base_url"],
        "num_concurrent": cfg.get("concurrent", 16),
        "max_retries": cfg.get("max_retries", 2),
        # The server tokenizes. Asking lm_eval to tokenize too means a second
        # tokenizer that can disagree with the one actually serving, and the
        # disagreement shows up as truncation that nobody ordered.
        "tokenized_requests": False,
        "max_length": cfg["max_length"],
    }
    args = ",".join(f"{k}={v}" for k, v in model_args.items())

    res = lm_eval.simple_evaluate(
        model="local-completions",
        model_args=args,
        tasks=cfg["tasks"],
        limit=cfg.get("limit"),
        num_fewshot=cfg.get("num_fewshot"),
        gen_kwargs=cfg.get("gen_kwargs") or None,
        log_samples=False,
        random_seed=cfg.get("seed", 0),
        fewshot_random_seed=cfg.get("seed", 0),
        numpy_random_seed=cfg.get("seed", 0),
        torch_random_seed=cfg.get("seed", 0),
    )
    # simple_evaluate returns numpy scalars and task objects that do not
    # serialise. Keep the scores and the config, drop the rest.
    slim = {
        "results": {k: {m: (float(v) if isinstance(v, (int, float)) else v)
                        for m, v in d.items()}
                    for k, d in (res.get("results") or {}).items()},
        "n-samples": res.get("n-samples"),
        "versions": res.get("versions"),
    }
    with open(out_path, "w") as fh:
        json.dump(slim, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
