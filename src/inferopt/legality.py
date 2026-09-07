"""Combinations vLLM rejects, checked BEFORE a launch is spent on them.

    for why in illegal(cfg):
        ...                       # skip it, or repair it, but do not launch it

Distinct from the flag check in evaluator, which asks whether vLLM ACCEPTS a
flag. These are flags it accepts individually and refuses together, which is the
failure an automated config generator actually produces.

The cost of not having this, measured: a Plackett-Burman screen puts
`max_num_batched_tokens` ON and `chunked_prefill` OFF in three of twelve design
rows by construction, so a quarter of every screen died. Worse, stage 2 pins
non-survivors by the sign of their effect estimate, pinned exactly that pair,
and every one of the eight factorial cells inherited it -- eight launches, no
answer, and the run reported no config at all.

Each rule cites the message vLLM produces, because a rule nobody can trace back
to an observed refusal is a guess that will one day be wrong in the other
direction and start rejecting valid configurations.
"""

from __future__ import annotations


def illegal(config: dict) -> list[str]:
    """Reasons this config cannot launch. Empty means no KNOWN violation.

    Deliberately not "is valid": these are the combinations we have watched
    fail. An empty list means nothing recognised, not a guarantee.
    """
    out: list[str] = []
    mnbt = config.get("max_num_batched_tokens")
    mml = config.get("max_model_len")
    chunked = bool(config.get("enable_chunked_prefill"))

    # vLLM: "max_num_batched_tokens (2048) is smaller than max_model_len (6144).
    # This effectively limits the maximum sequence length..." -- raised as a
    # SchedulerConfig validation error. Legal with chunked prefill on, because
    # then a long prompt is split across batches rather than needing to fit in
    # one.
    if mnbt is not None and mml is not None and not chunked and mnbt < mml:
        out.append(
            f"max_num_batched_tokens={mnbt} is below max_model_len={mml} with "
            f"chunked prefill OFF; a prompt must fit one batch, so vLLM rejects "
            f"this at SchedulerConfig validation")

    tp = config.get("tensor_parallel_size")
    if tp is not None and tp > 1:
        # vLLM shards attention heads across ranks and requires the division to
        # be exact. Not observed here -- inferopt has not run TP -- so it is
        # stated as a rule rather than quoted from a log.
        heads = config.get("_n_attention_heads")
        if heads and heads % tp:
            out.append(
                f"tensor_parallel_size={tp} does not divide the model's {heads} "
                f"attention heads")
    return out


def repair(config: dict, *, log=print) -> tuple[dict, list[str]]:
    """A launchable version of `config`, and what had to change.

    Repairs rather than rejects, because for a screening design the alternative
    is losing a whole row -- and losing rows is not neutral: it breaks the
    balance the estimates rest on, so the surviving effects become confounded
    with each other rather than merely with interactions.

    The repair is always the MINIMAL one that makes the combination legal, and
    it is always reported. A silently repaired config is a config the caller
    thinks it measured and did not.
    """
    cfg = dict(config)
    notes: list[str] = []
    for _ in range(4):                     # repairs can cascade; bounded anyway
        why = illegal(cfg)
        if not why:
            break
        mnbt, mml = cfg.get("max_num_batched_tokens"), cfg.get("max_model_len")
        if (mnbt is not None and mml is not None
                and not cfg.get("enable_chunked_prefill") and mnbt < mml):
            # Raise the token budget rather than switching chunked prefill on.
            # Turning it on would silently add a SECOND technique to the
            # configuration, and in a screening design that is the one thing
            # that must not happen -- the row would no longer measure what the
            # design says it measures.
            cfg["max_num_batched_tokens"] = mml
            notes.append(
                f"max_num_batched_tokens {mnbt} -> {mml} to clear max_model_len "
                f"with chunked prefill off")
            continue
        notes.append(f"unrepaired: {why[0]}")
        break
    if notes:
        log("        legality: " + "; ".join(notes))
    return cfg, notes
