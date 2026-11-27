# PACE

**Peer-wise Calibrated Experts against threat-knowledge drowning in benign-only IoT device onboarding.**

A newly deployed IoT device has only its own benign traffic. Peer devices hold labelled attacks and can share small supervised detectors. A common way to use them is to take the maximum of the peers' scores and put one threshold on it. That lets a single peer whose scores fire on the target's benign traffic raise the threshold above every other peer's attack evidence, so useful peers are suppressed ("drowning"). PACE keeps each peer as a separate expert, normalises and calibrates it on the target's own benign data, ORs the experts under a jointly calibrated false-alarm budget, and reserves part of the budget for a protected local branch, an anchored bank of benign-only experts fitted on the target's own traffic that adds evidence the peers cannot supply.

No target attack labels and no raw peer data are used. PACE offers no false-positive guarantee: the calibration rule is empirical, and realised false-positive rates are reported next to every detection result.

## Status

PACE is implemented end to end and validated on 58 devices from six IoT corpora (seven study strata) under strict provenance.

## Repository layout

```
src/pace/      types, config, core (numerics), datasets, experiment, reporting, cli
tests/         unit, property and architecture tests
rules/         Semgrep, Import Linter and Vulture configuration
configs/       the canonical configuration
data/          dataset links (raw/ is a symlink to a shared immutable directory)
results/       generated tables and figures
```

## Reproducing the validated experiment

Datasets are not stored in this repository; `data/raw` must point to the shared raw-dataset directory.

```bash
uv sync --locked
ln -s /path/to/shared/raw data/raw
uv run pace doctor
uv run pace preflight
uv run pace run all
uv run pace run iot_23_fresh
uv run pace report
uv run pace report --population confirmation
uv run pace accept
uv run pace accept --population confirmation
uv run pace audit
uv run pace benchmark
uv run pace calibration-study
```

The commands use `configs/default.yaml` (cross-fitted calibration, held-out evaluation, strict provenance); set `PACE_CONFIG` to use another configuration. `configs/development.yaml` is a development configuration whose output is never an official verdict. `pace run all` covers the seven study strata and `pace run iot_23_fresh` the independent confirmation corpus. `pace report` writes the tables, figures and `acceptance.csv` under `results/`; `pace accept` evaluates the same-calibration, fixed-FPR protocol and prints the verdict. For a clean-room run, point `runtime.output_root` at an empty directory.

### Exit codes

| Code | Meaning |
| ---- | ------- |
| 0 | success |
| 1 | gate not met or rejected (also incomplete sources, stale cells, role violations) |
| 2 | usage or configuration error |
| 3 | confirmation inconclusive because of a detection ceiling |
| 4 | runtime failure (message printed as `Error: ...`) |
| 130 | interrupted; completed cells are kept |

## Development

```bash
uv sync --locked
make check
```

`make check` runs ruff (lint and format), strict Pyright, Semgrep, Import Linter, Vulture, deptry and pytest with the 90 % coverage floor. CI runs the same gates and builds the package.

## License

MIT, see [`LICENSE`](LICENSE).
