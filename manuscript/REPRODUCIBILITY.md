# Manuscript reproduction

Run from the PACE project root with the existing locked environment. These commands read frozen evidence and generate manuscript summaries or publication figures only. They do not train, score or evaluate models again.

```bash
.venv/bin/python manuscript/scripts/verify_frozen_reporting.py --output-dir /tmp/pace-reporting
.venv/bin/python manuscript/scripts/verify_cache_diagnostics.py --output-dir /tmp/pace-cache-diagnostics
.venv/bin/python manuscript/scripts/generate_figures.py --output-dir /tmp/pace-figures
cd manuscript
latexmk -norc -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/pace-build main.tex
```

## Evidence and estimands

The reporting verifier validates all 110 canonical JSON content digests before aggregation. The unique key is (alpha, local fitting rows, study stratum, target, learning seed). All 408 repeated records must agree in every target field. Each attack type receives equal weight within target and seed; seeds 6, 7 and 8 are averaged, followed by equal target weighting. Targets share models and collection conditions; this does not establish independent deployment units.

The bootstrap independently reproduces the implementation: corpus order IOT_23, TON_IOT, GOTHAM_2025, CTU_13, N_BAIOT, UNSW_IOT_ATTACK_FLOWS; lexical target order within corpus; NumPy default_rng(314159); 10,000 within-corpus resamples; target-count weighting; linear 2.5% and 97.5% quantiles. Both UNSW strata belong to one corpus. Seeds and attacks are not resampled. Outcome-selected removals are exploratory. A singleton corpus contribution remains fixed.

For sweep adjustment, the verifier follows the explicit endpoint tests before interpolation, including the first-endpoint precedence when all stored FPRs tie. Replacing those tests with a single generic interpolation call can change tied-endpoint results. No matched-FPR outcome is newly measured.

## Generated material

- comparison-inventory.csv: 60 primary rows, comprising PACE, 12 study arms and two recorded baselines at both alpha levels and both fitting budgets. Recorded-baseline paired intervals and low-alpha FPR intervals are deterministic summaries of frozen rates.
- target-seed-rates.csv: all 21,240 method records across the 1416 unique target-seed-budget-alpha cells and 15 rules, including adverse outcomes and non-primary budgets.
- reporting-verification.json: primary TPR/FPR intervals, sweep coverage, gain concentration, selected deletion sensitivities and high-resource/seed-localized diagnostics.
- campaign-manifest.json: all frozen artifact paths, file SHA-256 values and embedded content/implementation digests.
- dataset-input-manifest.json: exact cache source paths, feature schemas and recomputed cache-manifest hashes. These hashes verify the manifests, not all raw archives.
- bibliography-records.json and bibliography-verification.json: primary DOI registry metadata and the mapping of all 37 final citations. Registry formatting is not blindly substituted for correctly published names, title acronyms or article numbering.
- source-access.json: additional institutional full text and policy/access boundaries.

## Figure sources and verification

1. architecture.pdf is drawn by the manuscript-only generator from the verified information flow in the methodology and scientific implementation; it has no experimental input.
2. target-gains.pdf uses results/tables/targets.csv for all four 58-target panels. Sorting is by gain, then stratum and target for deterministic ordering of tied values. No adverse target is removed.
3. false-alert-distribution.pdf uses target seed means from targets.csv for PACE and the primary comparator, and validated frozen campaign JSON for the recorded probability-max baseline. The standard ECDF assigns rank/58, with ranks 1 through 58; zero rates and the entire upper tail remain visible.
4. budget-scaling.pdf uses results/tables/scaling.csv for the fixed 58-, 47- and 15-target cohorts. Coordinates and confidence limits retain the stored six-decimal precision. The manuscript caption states the singleton-cohort conditioning.

figure-inputs.json archives all plotted numerical coordinates and source hashes. The manuscript figures reconstruct the publication layout because the original figure-specific source was absent. Their scientific values, labels, legends and tails were compared with the prior figures and frozen tables. Deterministic tie ordering can change coincident marker ranks without changing any target gain. No byte-identity claim is made for the earlier hand-supplied figures. Two separate runs of the new generator produced byte-identical PDFs after omitting variable PDF creation/modification dates; the numerical input archives also matched exactly.

The checked runtime is recorded in revision-evidence/runtime-versions.json. The existing uv.lock remains unchanged.

## Submission facts still requiring authors

Author identities, order, affiliations, all three emails, corresponding author and supplied ORCID are inserted and checked. Funding, competing interests, individual contributions, acknowledgements, final human responsibility/approval, exact private design/criterion/arm timing, missing raw-release retrieval/version records and unverified redistribution permissions require author confirmation. The public repository URL and immutable release revision are intentionally deferred.

The provisional Elsevier-style layout does not select a journal. The [Elsevier policy](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals) was checked on 21 October 2026: truthful AI assistance disclosure and human responsibility remain relevant. The manuscript retains its separate disclosure without asserting human approval that has not been supplied. Authors must check the eventual journal's requirements.

The canonical manuscript/audit.md contains the current verification statuses. The earlier verification.md is a historical preparation record.
