# Editorial review of PACE

Reviewed 22 October 2026. This is a wording, argument-precision and visual review of the current manuscript. It does not infer authorship from prose or certify an AI-detector result.

The paper's strongest claim is the measured benefit of replacing the complete local branch on this benchmark. Its presentation is stronger when it states that comparison directly, separates detection from false-alert costs, and explains where gains occur. The main editorial weakness was abstract, repetitive phrasing and numerically dense prose, rather than a list of supposedly AI-specific words. The changes below have been applied to the master.

## Findings and changes

| Priority | Finding | Applied improvement |
|---|---|---|
| High | The discussion described response-expert ablations as evidence of less extensive coverage. An ablation changes the calibrated final rule; it is not a direct coverage measurement. | Section 7.1 now says that removing the response expert causes smaller losses in final detection, while preserving the recalibration qualification. |
| High | Readers had to translate long decimal strings into practical changes and distinguish rates from differences. | Table 4 now labels rates as percentages and gains/intervals as percentage points. All eight rows were checked against unrounded reporting evidence. Selected ablation ranges and false-alert changes use readable percentage-point values; the full numerical inventory remains available. |
| Medium | Abstract and introduction relied on abstract framing such as an asymmetric learning problem and architectural tradeoffs. | Recast the opening around the actual data available to a target, what peers contribute, and when local detectors add detections. The conclusion identifies the next evidential question: whether gains persist on untouched traffic. |
| Medium | Repeated qualifications and self-referential transitions interrupted the argument. | Shortened transitions and repeated cautions while retaining the limits on calibration, attribution, cohort reuse and deployment claims. Results headings now name the analysis directly. |
| Medium | Figure 1 interrupted the method explanation; a float barrier could strand the false-alert paragraph away from its table. | Placed Figure 1 with the method overview and kept the false-alert subsection, its opening paragraph and Table 5 together. |
| Low | A lone continuation word, split bibliography entries and a short appendix paragraph start weakened page flow. | Added widow/orphan protection, kept bibliography entries together and moved the appendix paragraph to a page with adequate room. |
| Low | A reference to a manuscript supplement did not identify the actual evidence file. | Appendix B now gives the dataset-input manifest's path; other appendix and section references are explicit. |

## Representative wording improvements

- Abstract: “An Internet of Things (IoT) target may have trusted benign observations but no labeled attacks, while supervised models trained at its peers provide attack evidence.” This establishes the information regime immediately.
- Discussion: “Removing the response expert causes smaller losses in final detection.” This describes the measured outcome precisely.
- Conclusion: “Validation on untouched traffic is needed to determine whether the detection gains persist and false-alert rates are acceptable in deployment.” This replaces a generic call for future validation with a concrete test.

## Remaining reviewer concerns

1. **External validity is the main scientific weakness.** The benchmark informed development. Further polishing cannot turn the evaluation into independent validation. An untouched traffic evaluation would strengthen the paper more than additional stylistic edits.
2. **The improvement is heterogeneous.** Gains concentrate in TON_IoT and a small number of targets, while many targets change little. Keep the target-level plot and exclusion analyses; avoid describing the mean gain as improvement for most devices.
3. **Operating-point fairness remains unresolved.** Nominal-alpha comparisons and clamped coarse sweeps do not establish general superiority at equal realized FPR. A future matched-operating-point evaluation would address this, but no new experiment was run in this pass.
4. **Operational false alerts remain substantial.** The worst-target FPR is about 0.67. Lower mean FPR in the alpha=0.05 comparison is useful evidence, but does not establish acceptable deployment workload.
5. **Some results paragraphs remain numerically dense.** Their detailed intervals and adverse cases are scientifically useful. If a chosen journal requires a shorter paper, move secondary subgroup details to an appendix while retaining the conclusions and explicit limitations in the main text.

## Verification

The revised PDF has 31 pages, with the conclusion on page 25. All pages were visually checked, including the four figures, appendix tables and bibliography. The LaTeX build has no warnings or overfull/underfull boxes. Main prose decreased from 8,631 to 8,551 TeXcount words in this pass; the abstract contains 222 counted prose words. All eight reformatted primary-table rows agree with the existing unrounded reporting evidence. Display equations in edited sources are unchanged. All 175 protected repository files match their recorded hashes. Scientific code, frozen results and figure data were not modified; no training or scientific campaign was rerun.

This pass improves readability and precision. It does not increase the strength of the underlying evidence or guarantee acceptance.
