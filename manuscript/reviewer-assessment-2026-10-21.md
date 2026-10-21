PACE manuscript: independent reviewer assessment

Date: 21 October 2026. Overall grade: **76/100**. Recommendation: **major revision before journal submission**.

This is a subjective research-quality assessment under a general applied intrusion-detection journal standard, not an acceptance prediction or a grade against a particular journal's rubric. The manuscript is technically careful and unusually transparent. Its main weakness is the distance between a well-documented architectural benchmark and independent evidence that the proposed complexity delivers a robust, practically useful advantage.

The previous audit's statement that the manuscript is ready for author approval is compatible with its editorial completion, but does not establish that its scientific evidence is sufficient for acceptance. This assessment evaluates that separate question. Most major concerns below are already disclosed accurately in the manuscript; disclosure earns credit for honesty but does not resolve the underlying evidence gap.

| Dimension | Score | Reason |
|---|---:|---|
| Contribution and originality | 14/20 | A coherent combination of retained peer decisions and complementary local evidence; limited isolation of the motivating failure mechanism and limited evidence that every tier is needed. |
| Experimental evidence | 20/30 | Six corpora, 58 targets, paired comparisons and extensive ablations; reused development cohort, internal comparators, concentrated gains and incomplete FPR matching limit the conclusions. |
| Methodological specification and claim discipline | 17/20 | Equations, data roles, implementation conventions and limitations are explicit; empirical calibration, dependent targets and preprocessing complicate interpretation. |
| Reproducibility and reporting | 13/15 | Frozen-result calculations reproduce and comparison inventory is complete; public release identifier and some raw-data provenance remain pending. |
| Writing and presentation | 12/15 | Clear technical prose and readable figures; repetitive qualifications, dense numerical narration and implementation-history material obscure the main scientific story. |
| **Total** | **76/100** | **A credible research draft requiring stronger validation.** |

Evidence checked

- Read the full manuscript's LaTeX sections and tables, including both appendices and declarations; inspected selected core, experiment and reporting implementation paths.
- Rendered all 35 pages of the delivered PDF and reviewed their layouts in contact sheets, with separate inspection of the primary-results table and target-gain/FPR figures. No obvious clipping, overlap or missing visual elements was found in that inspection.
- Built the manuscript afresh in a temporary directory. The final LaTeX log contains no unresolved warnings, undefined references, or overfull/underfull boxes. BibTeX reports an empty-pages warning for the Kitsune entry; check its published citation format rather than inventing pages to silence the warning.
- Reran the frozen-reporting verifier: 110 campaign artifacts, 1,824 records, 1,416 unique records and 408 identical repeats. It validates canonical content digests before aggregation.
- Independently compared all 60 numerical appendix rows against the newly recomputed comparison inventory; TPR, FPR, differences and interval endpoints agree to printed precision.
- Ran the existing core tests, reporting-metric tests and experiment-arm tests: **85 passed**. These tests support implementation consistency; they do not validate deployment generalization.
- Spot-checked the closest MCDE comparison against the supplied full text and its publisher search record, and located the primary Arp evaluation-pitfalls publication. This was not an exhaustive citation-by-citation literature audit.

No scientific campaign was rerun, no new benchmark was trained, and no raw-data reconstruction was completed. The manuscript, implementation and frozen scientific results were not edited. Temporary verification outputs are under `/tmp/pace-independent-audit-20261021` in Ubuntu. This report is the only intended repository addition.

What is already strong

The primary effect is reproducible from the stored outputs. The most persuasive result is at nominal alpha = 0.05: the complete local-branch replacement improves average detection while reducing average false alerts. At N = 400, detection increases from 87.36% to 89.68%, and mean FPR falls from 6.44% to 5.43%. That is a meaningful comparative result on this benchmark, even though it does not imply joint improvement on every target.

| Nominal alpha | Local fitting rows | Detection gain, percentage points | Mean FPR change, percentage points |
|---:|---:|---:|---:|
| 0.01 | 100 | +2.609 | +0.348 |
| 0.01 | 400 | +2.831 | +0.159 |
| 0.05 | 100 | +2.342 | -0.198 |
| 0.05 | 400 | +2.322 | -1.007 |

All four reported paired detection intervals exclude zero under the stated corpus-stratified bootstrap. The paper properly limits the interpretation of those intervals. It also reports adverse arms, target regressions, concentration of gains, unequal total data access, and FPR tails instead of hiding them. The distinction between trusted benign target data and unseen attack families is particularly important and is handled well. The equations and pseudocode broadly align with the inspected code, including strict exceedances, higher quantiles, reference-tail extrapolation and the low-alpha allocation exception.

Major findings, in priority order

1. **The evaluation does not independently validate the developed architecture.** Location: Section 5.5, `sections/experimental-methodology.tex:50`, and Section 8.

   The benchmark cohort informed development, and a failure on IoT-23 capture 35-1 motivated the anchor resolution rule. New learning seeds on the same targets do not remove selection through earlier architectural decisions. The recorded commit sequence cannot establish when outcomes were first inspected. The present disclosure is appropriate, but it leaves a substantial generalization gap.

   Required strengthening: freeze the architecture, preprocessing, comparators and evaluation criteria before evaluating untouched captures, devices or a genuinely unexamined corpus. A chronological external campaign would be particularly valuable. Merely recomputing leave-one-stratum-out summaries from this campaign is not a fresh validation experiment. Resolve the private freeze-history statement factually; do not retroactively describe the protocol as preregistered.

2. **The peer-wise contribution remains confounded with operating point.** Location: Sections 6.2 and 6.6, `sections/results.tex:21` and `sections/results.tex:75`.

   With the complete local branch fixed, separate peer decisions improve detection over mean/max fusion, but also increase mean FPR. For example, at alpha = 0.01 and N = 100, PACE versus peer-max raises TPR from 78.785% to 83.340%, while FPR rises from 2.512% to 3.399%. Those observations establish a tradeoff, not equal-FPR superiority.

   The primary sweep adjustment also clamps outside coverage in 75/174, 69/174, 36/174 and 28/174 target-seed cells, approximately 43.1%, 39.7%, 20.7% and 16.1%. At those cells, it does not estimate performance at a genuinely matched FPR. The manuscript explains this correctly, but the adjustment cannot close the inferential gap.

   Required strengthening: evaluate sufficiently dense operating curves for PACE and the actual peer-fusion comparators; report covered regions and target-level tradeoffs. Use held-out benign calibration for deployable threshold selection. Treat any operating point selected using test FPR as descriptive/oracle analysis, not as a deployment procedure. A covered-only sensitivity summary of existing sweeps can help, but must disclose the changed subset and cannot replace the missing measurements.

3. **The proposed architecture is not yet convincingly justified against simpler and external alternatives.** Location: Sections 2.1, 5.3, 6.1 and 6.3; Appendix A.

   The strongest headline comparator shares PACE's peer branch and changes the complete local branch. It therefore answers an internal design question. MCDE is discussed but not executed; the paper does not provide a common-protocol external performance comparison. This is a gap in validation, not a false claim of state-of-the-art superiority.

   Within the existing results, all four detection intervals against peers-only include zero. At low alpha, the single-local-union arm matches PACE's pooled TPR to stored precision and has nearly identical FPR. At alpha = 0.05, bank-only has higher detection and higher FPR. These are credible alternatives, not peripheral ablations. The response expert's modest effect further raises a complexity-versus-benefit question.

   Required strengthening: compare against a small number of relevant, reproducible external and simpler baselines under the same allowed information. Any MCDE adaptation must explicitly preserve the benign-only target restriction and distinguish it from the original protocol. Add a controlled six-member mean comparator and a version with comparable calibration to separate detector composition from thresholding. Measure inference cost if complexity is part of the practical justification. Do not compare headline numbers copied from other papers.

4. **False-alert behavior limits operational value, even where relative performance improves.** Location: Section 6.5 and Table 5.

   At alpha = 0.01, mean FPR is approximately 3.24-3.40%, worst-target FPR is approximately 67%, and around 26% of targets exceed twice the nominal parameter. At alpha = 0.05, the 90th percentile remains approximately 16%. These are serious tail failures for a detector intended to operate on predominantly benign traffic. The primary comparator shares the problem, so this is not evidence that PACE alone causes it.

   Required strengthening: identify the failing targets and responsible branches, distinguish calibration-role behavior from later-test failure, and evaluate any proposed repair on untouched data. Report total benign onboarding requirements and, if making operational claims, alert aggregation or incident burden. A new FPR theorem is not mandatory for an empirical systems paper; convincing validation of a useful operating regime is.

5. **The calibration exception has weak necessity evidence.** Location: Section 4.5, `sections/methodology.tex:84`.

   At alpha = 0.01, the peer, anchor and bank allocations sum to 0.0125. The threshold-resolution check uses the configured block parameter rather than every block's actual length. This is accurately described as an implementation heuristic, so it is not a hidden mathematical error. Nevertheless, the allocation-respecting alternative attains the same pooled detection at this setting. The current evidence does not demonstrate why the exception deserves to be central to the method.

   Required strengthening: directly compare allocation-respecting and exception-based rules, with calibration alert counts, target outcomes and equal-FPR sensitivity. Prefer the simpler rule if no reliable advantage survives independent validation. Changing the algorithm would require new results and a clear distinction from the present campaign.

6. **Effective evidence is narrower than the target count suggests.** Location: Sections 5.4, 6.4 and 6.7.

   The three highest-gain targets account for approximately 60-89% of summed gains. Excluding TON_IoT reduces average gains to approximately 0.55-1.02 percentage points. At N = 400 and alpha = 0.05, 15 targets improve, 41 are effectively unchanged and two regress under the declared 0.001 tolerance. Target reuse of peers and collection conditions also means that 58 is not a count of independent deployments.

   Required strengthening: foreground per-corpus and target-level effect distributions, including the two-dimensional TPR/FPR tradeoff. Retain the existing paired target analysis, but avoid treating its confidence intervals as population-level confirmation. A dependence-aware sensitivity analysis may be informative; with only six corpora, a corpus-level bootstrap is itself limited and is not an automatic statistical cure. Generalization needs additional independent collection conditions.

7. **The benchmark's information flow constrains its onboarding interpretation.** Location: Sections 3, 5.1 and 5.2.

   N counts local detector fitting rows; pooled standardization sees complete benign training roles, and reference/calibration rows are additional. Gotham's full-second features can share contributors across roles and include later or malicious packets in a nominally benign row. Client/label-specific chronological splits also do not impose a global network cutoff. These are disclosed protocol properties, not grounds for alleging direct use of target attack labels in fitting.

   Required strengthening: retain the current narrow wording. For practical onboarding claims, evaluate source-frozen or strictly permitted preprocessing, causally available features, temporally grouped boundaries and explicit total data requirements. A Gotham exclusion sensitivity could assess its contribution, but does not repair its feature construction.

8. **Peer drowning is a plausible explanation that has not been isolated experimentally.** Location: Introduction and Sections 6.2 and 7.2.

   Aggregate gains from separately thresholded peers can arise from several changes in the decision rule. They do not show that one peer's benign tail specifically suppresses another peer's useful attack evidence. The manuscript itself acknowledges that distinction.

   Required strengthening: add a diagnostic experiment that measures peer-specific benign tails, thresholds and recovered detections when a dominating peer is introduced, removed or perturbed, while holding the local branch fixed and accounting for false-alert costs. Alternatively, present peer drowning as motivation and keep the empirical contribution centered on measured complementary coverage. The present cautious prose is much better than asserting a proven mechanism.

Editorial and presentation findings

- **Condense the repeated qualifications without removing them.** Related work, results, discussion and limitations repeatedly restate absent guarantees and unavailable claims. Keep qualifications next to the claims that need them, then consolidate implementation-history and verification detail in the supplement. A roughly 15-20% reduction in main-text prose is a reasonable editing target, not a venue-imposed requirement.
- **Reduce numerical density.** Table 4 and several result paragraphs use six-decimal proportions where readers need a few percentage-point effects and intervals. Keep full precision in machine-readable artifacts and use consistent, readable rounding in narrative text.
- **Make the best-supported result the organizing claim.** The local-branch gain at alpha = 0.05, with lower mean FPR, is more compelling than a general superiority story about peer calibration. Present peer fusion as a second question with an explicit alert-rate tradeoff.
- **Clarify an ambiguous evidence link in Section 7.2.** `sections/discussion.tex:13` discusses peer-fusion comparisons and immediately refers to clamped sweep estimates. Section 6.6 defines the primary clamped comparison against the mean-local comparator. Identify the comparator explicitly so the reader does not infer that these estimates directly establish matched-FPR gains against fixed-local peer-mean/max arms.
- **Use consistent branch terminology.** `sections/experimental-methodology.tex:32` says fused-peer arms retain the local bank; the abstract, results and code correctly establish that the complete local branch, including the anchor, is fixed. Use the complete description in Section 5.3 as well.
- **Improve low-range figure readability.** Figures 2 and 3 retain the full tails, which is good. A clearly labeled near-zero inset would help readers inspect small target gains and the operational FPR range without hiding the adverse extremes.
- **Complete submission facts.** The release URL/revision, funding/conflicts, author contributions, final responsibility/approval and unresolved provenance facts remain pending. Preserve truthful AI-assistance disclosure and obtain the actual author declarations; these are separate from scientific validation.

Recommended revision sequence

1. State the narrow supported contribution and freeze the next evaluation protocol. Resolve the factual design-history and release requirements.
2. Evaluate fair operating curves and a concise set of simpler and external baselines under identical information access. Diagnose the FPR tail failures.
3. Validate the frozen method on untouched data with suitable chronological and preprocessing separation. Additional seeds on the same cohort are insufficient for this step.
4. Use the new evidence to decide whether the anchor exception and response branch are worth retaining. Present target-level utility and computational cost.
5. Tighten the main text and complete publication declarations after the scientific conclusions are settled.

If new experiments are unavailable, the defensible paper is an exploratory architectural study on these six corpora. Its reproducible modest gains and candid failure analysis still have value. Editing can improve clarity and claim discipline, but cannot provide independent validation or establish equal-FPR superiority.

Selected external sources consulted

- [Yue, Lu and Xie, Detecting Unseen IoT Attacks with Calibrated Dual Evidence Under Low False-Positive Budget](https://www.mdpi.com/1099-4300/28/9/1026). Closest hybrid-calibration comparator; the supplied full text was used because direct publisher fetching was rate-limited. It supports the architectural comparison, not a numerical ranking against PACE.
- [Arp et al., Pitfalls in Machine Learning for Computer Security](https://www.mlsec.tu-berlin.de/docs/2024-cacm.pdf). Primary-author copy supporting the relevance of evaluation-design and data-selection concerns. The corresponding earlier conference paper is available from [USENIX](https://www.usenix.org/conference/usenixsecurity22/presentation/arp).
