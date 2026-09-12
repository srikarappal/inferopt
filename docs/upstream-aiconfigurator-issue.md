# Relax `numpy~=1.26.4` and bound `plotext`: two metadata issues blocking installation alongside modern stacks

## Summary

`aiconfigurator` 0.11.0 and `aiconfigurator-core` 0.11.0 both declare
`numpy~=1.26.4`, which is a hard upper bound of `<1.27`. This makes the package
uninstallable alongside anything that requires numpy 2.x without breaking one
side or the other, and numpy 2.0 has been out since mid-2024, so most of the
surrounding ecosystem has moved.

Testing suggests the bound is not needed. Every public entry point produces
**byte-identical output** under numpy 1.26.4 and numpy 2.3.5.

Separately, and in the opposite direction, `plotext>=5.3.2` is too loose:
plotext 6 removes an attribute the CLI path calls, so a fresh install that
resolves plotext 6 fails at runtime.

## The numpy bound

Declared in both distributions:

```
aiconfigurator      0.11.0   Requires-Dist: numpy~=1.26.4
aiconfigurator-core 0.11.0   Requires-Dist: numpy~=1.26.4
```

`~=1.26.4` means `>=1.26.4, <1.27`. Consequences for anyone installing into an
environment that already uses numpy 2:

- `pip install aiconfigurator` silently **downgrades** numpy to 1.26.4, which
  breaks any installed package built against numpy 2 ABI.
- Constraining numpy to keep it, `pip install aiconfigurator -c "numpy>=2"`,
  fails with `ResolutionImpossible`.
- Asking for both at once is worse than either. Given `aiconfigurator` with
  numpy constrained to 2.x, pip backtracks and silently resolves
  **aiconfigurator 0.1.1**, a much older release, pulling in gradio and
  downgrading pydantic and pandas, while reporting a successful install. Pinning
  `aiconfigurator>=0.11` turns that into a clear `ResolutionImpossible` instead,
  which is at least honest, but still leaves no way to install.

Every other dependency the package declares is a `>=` floor, so numpy is the
only upper bound in the metadata.

## Evidence that numpy 2 works

Two environments, identical apart from numpy. All five public entry points from
`aiconfigurator.cli`, same model, same system, same arguments.

| entry point | numpy 1.26.4 | numpy 2.3.5 |
|---|---|---|
| `cli_support` | `agg_supported=True, disagg_supported=True, exact_match=False, architecture='Qwen3ForCausalLM', agg_pass_count=18/18, disagg_pass_count=18/18` | identical |
| `cli_generate` | `{'gpus_per_worker': 1, 'gpus_used': 8, 'pp': 1, 'replicas': 8, 'tp': 1}` | identical |
| `cli_default` | 4 rows; top: `ttft 187.127, tpot 18.023, tokens/s/gpu 27787.234, tokens/s/user 55.485, tp 1, pp 1, bs 512, concurrency 512` | identical |
| `cli_recommend` | `agg: total_gpus_needed 7, replicas_needed 7, tp 1, tpot 28.858` / `disagg: total_gpus_needed 12, replicas_needed 3, tpot 29.121` | identical |
| `cli_exp` | 0 rows for the given config | identical |

A `diff` of the two JSON dumps, with the numpy version line removed, is empty.

Reproduction:

```bash
python -m venv env && ./env/bin/pip install aiconfigurator          # numpy 1.26.4
./env/bin/python probe.py > np1.json

./env/bin/pip install --no-deps --upgrade numpy==2.3.5              # force numpy 2
./env/bin/pip install --no-deps "plotext<6"                         # see below
./env/bin/python probe.py > np2.json

diff <(grep -v numpy np1.json) <(grep -v numpy np2.json)            # empty
```

Where `probe.py` calls `cli_support`, `cli_generate`, `cli_default`,
`cli_recommend` and `cli_exp` and dumps the fields above as sorted JSON.

Caveat on scope: this is one model architecture on one system, exercising the
five public entry points. It does not run the project's own test suite under
numpy 2, which would be the stronger check and is presumably easy for a
maintainer to do in CI.

## The plotext bound, in the other direction

`plotext>=5.3.2` permits plotext 6, and a fresh install resolves it. `cli_default`
then fails:

```
AttributeError: module 'plotext' has no attribute 'plot_size'
```

`plot_size` was removed in plotext 6. So the package currently declares a floor
where it needs a ceiling, and a fresh `pip install aiconfigurator` followed by
`cli_default(...)` fails on a default resolution. Pinning `plotext<6` fixes it.

## Suggested change

```diff
- numpy~=1.26.4
+ numpy>=1.26.4

- plotext>=5.3.2
+ plotext>=5.3.2,<6
```

in both `aiconfigurator` and `aiconfigurator-core`.

The numpy change makes the package installable next to modern numpy without any
`--no-deps` gymnastics, and the plotext change makes a default install work
rather than failing on the first `cli_default` call. If a numeric path does
depend on numpy 1 behaviour that the five entry points above do not reach, a
narrower bound naming that reason would still be far more usable than `<1.27`.

Happy to open a PR for either change if that is useful.
