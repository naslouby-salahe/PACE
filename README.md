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
uv run pace report
uv run pace accept
```

The commands use `configs/default.yaml`. `pace run all` writes each stratum's figures to `outputs/figures/`, `pace report` builds the project's `results/` folder (CSV and LaTeX tables, PDF and PNG figures) and `pace accept` prints the acceptance verdict. Set `PACE_CONFIG` to use another configuration; use an empty `runtime.output_root` for a clean-room run.

## Development

```bash
uv sync --locked
make check
```

`make check` runs ruff (lint and format), strict Pyright, Semgrep, Import Linter, Vulture, deptry and pytest with the 90 % coverage floor. CI runs the same gates and builds the package.

## License

MIT, see [`LICENSE`](LICENSE).
