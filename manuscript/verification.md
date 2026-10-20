# PACE revision audit

Revised 13 October 2026. **Editorial draft complete; author confirmation required before submission.** Governing request: attachment 48c252f8-fab8-4454-a1c9-bb65a59bc886/pasted-text-1.txt.

## Resolved

- **Baseline and scope:** inspected the matching 26-page manuscript, LaTeX, bibliography, documentation, executed detector/calibration/preprocessing/statistical code, configuration and frozen results. Baseline title, figures, tables and structure matched the supplied description. Initial working tree was clean. Changes are confined to manuscript/: no scientific code, tests, configuration, datasets or frozen results changed; no retraining, campaign reruns, commits or pushes.
- **Title, abstract, introduction and conclusion:** adopted requested title and evidence-led summaries. Primary gains and favorable alpha 0.05 detection/FPR outcomes are prominent within the first two pages. Removed adjacent architectural repetition. Abstract has 209 whitespace-delimited words; eventual journal limit remains an author-dependent check.
- **Related work:** preserved substantive comparisons and rebuilt Table 1 as two readable panels containing all five requested fields. MCDE precedent, retained foreign peers, target information, separate calibration and final OR are explicit; no external performance ranking or new-estimator/theorem claim.
- **Problem/method:** added data-access Table 2, vector architecture Figure 1 and Algorithm 1. Verified ties, extrapolation, strict exceedance, balanced blocks, higher quantiles, shared multiplier, feasible-cap return and 20-bisection behavior, anchor heuristic and final OR. Alpha .01 allocations remain .005/.005/.0025 (sum .0125); alpha .05 remains .025/.0125/.0125.
- **Experimental methods:** identified all shared comparator components; retained 40/20/20/20 roles, extra benign data and pooled full-training preprocessing, seeds 6/7/8, 10,000 bootstrap replicates/seed 314159, fixed corpus interpretation, exploratory intervals, unchanged cohorts and engineering-gate/planning-power definitions.
- **Results:** ordered primary outcomes, complementarity, corpus/target variation, FPR, clamped sensitivity, budget scaling and seed/protocol checks. Table 4 pairs exact detection and both FPR means; Table 5 preserves all distribution metrics.
- **Figures:** four vector PDFs. Figures 2/3 use full-precision frozen campaign values, enlarged panels, all 58 targets/configuration, negative points/outliers, original three FPR methods and nominal guides. Marker shapes/line styles supplement color. Figure 4 uses unchanged frozen scaling estimates/intervals, distinguishes 58/47/15-target cohorts and represents duplicate primary/robustness points for the identical 58-target cohort once. Connecting segments are visual guides.
- **Discussion/limitations:** four interpretive subsections and five threat categories. Local-exclusive coverage is distinguished from comparator gain; peer-fusion ablations separately address retained peer evidence.
- **References:** independently checked 29 DOIs (all HTTP302), Crossref metadata for 27 scholarly records, DataCite for two datasets, and claim-specific original content. Corrected Zimek's omitted subtitle to Ensembles for Unsupervised Outlier Detection: Challenges and Research Questions [Position Paper]. No entries added/removed. Inventory: 26 peer-reviewed publications, one explicitly labeled preprint, two datasets. No SciSpace used.
- **Presentation:** retained journal-neutral Elsevier-compatible elsarticle and removed the unselected journal name. Author placeholders and AI-use disclosure remain. Elsevier's [AI policy](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals) was checked; completed human approval is not falsely asserted.

## Numerical verification

Independent Reviewer B recomputed primary means and corpus-stratified bootstrap intervals from frozen campaign JSON. Order: (.01/100), (.01/400), (.05/100), (.05/400).

| Quantity | Verified values |
|---|---|
| PACE detection | .833405 / .838987 / .894767 / .896837 |
| Comparator detection | .807318 / .810681 / .871348 / .873619 |
| Paired gains | .026086 / .028306 / .023418 / .023219 |
| 95% intervals | [.011323,.043346] / [.012339,.047375] / [.005777,.043968] / [.003116,.045968] |
| PACE mean FPR | .033992 / .032369 / .055453 / .054337 |
| Comparator mean FPR | .030516 / .030781 / .057437 / .064407 |
| Clamped gains | .024659 / .025735 / .022176 / .020718 |
| Clamped intervals | [.009682,.042034] / [.009690,.044983] / [.005135,.042086] / [.002056,.042482] |
| Endpoint cells /174 | 75 (47 below,28 above) / 69 (49,20) / 36 (12,24) / 28 (9,19) |
| Excluded-TON gains | .009306 / .010228 / .005536 / .008299 |
| Excluded-TON intervals | [.001808,.019526] / [.000995,.024240] / [.000090,.014343] / [.000143,.022250] |
| Rarity removal costs | .008626 / .011977 / .006020 / .010403 |

All ablation values, peer-fusion gains/FPR ranges and eight positive intervals agree with ablations.csv. Stratum, target, decomposition, seed and budget claims agree with frozen tables. Preserved CTU-13 scenario-12 loss .032595; worst PACE FPR .669005/.670842 at alpha .01 and .671844 at .05; local-exclusive .147434/.120957/.000042 for TON_IoT/IoT-23/N-BaIoT at .01/400; high-resource alpha .05 budget decline .028296 with interval [-.032827,-.023766]. No new matched-rate estimate or inferential test.

## Adversarial reviews and repeat checks

| Review | Resolved | Confirmed limitation |
|---|---|---|
| A: novelty/literature | Independent metadata/original-content audit; clarified precedent and five table fields; repaired title. Lead repeated final related-work/table/bibliography review. | Integration may be judged incremental; no executed external competitors. |
| B: statistics | Independent recomputation of primary, clamped, excluded-TON intervals and FPR; verified adverse results, cohorts, ablations and code. Final lead reread confirmed numerical sections unchanged after audit. | Dependence, development reuse, three seeds, multiplicity, endpoint clamping. |
| C: security | Independent own-peer exclusion, pooled scaling, role-access, Gotham and chronology checks; repeated compiled-text check found no new overclaim. | Trusted benign onboarding/peers; mixed-second dependencies; no operational validation. |
| D: presentation | Independent first-build inspection of all pages identified floating connector, unsupported bold small caps and introductory repetition; all fixed. Lead inspected every final page and new Table 1. | Author-dependent fields remain pending. |

The literature agent hit a service usage limit after saving its full audit; the lead completed the final repeat directly. Not every repeat was performed by a separate agent. Further rewriting stopped after supported corrections and final visual verification to avoid churn.

## Confirmed limitations

Nonrandom/development-used targets and shared peers/captures; conditional bootstrap and exploratory multiplicity; trusted target benign data beyond N and target-influenced pooled scaling; compatible schemas and labeled peer attacks; Gotham mixed-label complete-second aggregation and no network-wide prospective cutoff; empirical allocation/extrapolation/final OR without future FPR guarantees; heterogeneous alerts and target regressions; incomplete expert-level attribution and external baselines; nonmonotonic budgets; no tested malicious-peer, poisoning, evasion, privacy, latency, energy, communications or online-deployment guarantees. These are evidence limitations, not problems cured by rewriting.

## Blocked author-dependent submission items

Journal choice/exact abstract and format requirements; names, affiliations, correspondence and CRediT; funding/conflicts/acknowledgements; public code archive/release identifier and dataset redistribution conditions; prior/overlapping-publication disclosure; final human scientific approval and AI-assistance responsibility. None invented. These do not block delivery of the requested final draft; they block submitting it unchanged.

## Final build and integrity

TeX Live/latexmk/BibTeX build: 28 pages, four figures, five tables, Algorithm 1, eight equations, 29 references. Final log has no warnings, missing citations/references, overfull or underfull boxes. All final rendered pages inspected; no clipped text/legends/axes/glyphs, table collisions or orphan headings. Enlarged plots use the available width; ordinary terminal bibliography whitespace remains.

Deterministic Unicode inspection: zero suspicious characters across all 18 manuscript text/source files. All 29 bibliography entries are cited, with no missing keys.

PDF SHA256: 9d2b28b22dfec48a31af4c6b0e8573c344adeb6a76581d28f884fd71b48bdf89

Source structure preserved; temporary evidence/build material moved outside repository. No permanent analysis scripts or research infrastructure. Final Git check confirms only manuscript changes. Numerical checks read frozen outcomes without experiments.

## Reference source register

Original content below was independently inspected for attributed claims; DOI resolution alone was not treated as content verification. Publication metadata remains in references.bib.

| Key | Primary/original source; inspected support |
|---|---|
| yue2026mcde | https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13605448/fullTextXML : §§3-4 benign-tail branches, budget-adjusted fusion, second disjoint threshold set, HGB, family holdout and benign device-transfer subsection. Supports MCDE row and rigorous distinction. |
| barber2023nonexchangeable | https://arxiv.org/pdf/2202.13415 : introduction and §3.1 weighted quantiles, distribution drift/exchangeability, nonsymmetric algorithm extension. Supports bounded conformal comparison. |
| bates2023outliers | https://arxiv.org/pdf/2104.08279 : marginal/conditional validity definitions and dependence from shared calibration. Supports distinction from PACE empirical calibration. |
| angelopoulos2023conformal | https://arxiv.org/pdf/2107.07511 : finite-sample ceiling correction, calibration/test interpretation, exchangeability and shift subsections. Supports conventional split-conformal comparison. |
| liu2008isolation | https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf : random partition isolation and short average path lengths. |
| breunig2000lof | https://sigmodrecord.org/?download_id=6631&smd_process_download=1 : complete title/authors and local reachability density definition. |
| pang2021review | https://arxiv.org/pdf/2007.02500 : reconstruction-based representation versus anomaly scoring discussion; autoencoder reconstruction objective. |
| diot2019 | https://arxiv.org/pdf/1804.07474 : device-type GRU probabilities, benign model assumptions, collaborative federation, low-probability packet definition and window trigger; §VI-A threshold study uses benign Activity and Attack datasets. |
| kitsune2018 | https://arxiv.org/pdf/1802.09089 : ensemble errors into output AE, train/exec mode, assumed normal initial traffic, discussion of maximum training score or probabilistic cutoff and application-dependent selection. |
| nbaiot2018 | https://arxiv.org/pdf/1805.03409 : device-local AE, benign training/optimization sets, MSE threshold and majority window settings, nine devices and Mirai/BASHLITE context. |
| osdn2024 | https://arxiv.org/pdf/2311.11249 and https://arxiv.org/html/2311.11249v1 : §3 labeled source, unobserved target ground truth, unknown target categories, distinct feature dimensions; shared classifier with generated unknown examples. |
| talpini2025 | https://marcosavi.altervista.org/Papers/2025_ICAART_Federated_Calibration_IDS.pdf : Platt sigmoid parameters, labeled input-output calibration data, cross-entropy, federated calibrator training. |
| vim2022 | https://arxiv.org/pdf/2203.10807 : feature residual converted to virtual logit and combined with classifier logits; vision setting. |
| zimek2014 | https://www.kdd.org/exploration_files/V15-01-02-Zimek.pdf : title/subtitle, §4.2 normalization, maximum fusion discussion and individual overestimation. |
| hyperids2025 | https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12485126/fullTextXML : hypernetwork and client embeddings, categorical cross-entropy, labeled classes. |
| ecod2022 | https://arxiv.org/pdf/2201.00382 : Algorithm 1 and eqs4-6, three aggregate left/right/skewness scores and maximum; supports explicit distinction from PACE featurewise-max sum. |
| jin2026crossprotocol | https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12944665/fullTextXML : unlabeled target UDA, random windows versus capture-wise/LOCO, source-selected threshold false-alert failure. |
| garcia2020iot23 | https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/ : original dataset description, 20 infected captures and 3 benign captures; current PACE subset count derives from repository, not original source. |
| belarbi2025gothamdata | https://zenodo.org/records/14502760 : current full title, all six creators, published date/version, 78 device-level virtualized smart-city traces and ground-truth labeling description. |
| vovk2020averaging | https://sas.uwaterloo.ca/~wang/papers/2019Vovk-Wang-BIOM.pdf : p-value input premise, arbitrary dependence, Bonferroni and mean scaling. |
| sommer2010closed | https://www.icir.org/robin/papers/oakland10-ml.pdf : semantic gap from anomalous input to operational security meaning. |
| saito2015precision | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432 : class-imbalance discussion and prevalence-dependent PR baseline. |
| arp2024pitfalls | https://www.mlsec.tu-berlin.de/docs/2024-cacm.pdf : sampling bias and laboratory-performance pitfalls in security ML. |
| hamza2019mud | https://www.contrib.andrew.cmu.edu/~theophib/papers/SoSR19.pdf : original title uses IoT (Crossref typo loT ignored), one-minute flow counters and MUD-flow modeling. |
| garcia2014ctu | https://www.researchgate.net/profile/Sebastian-Garcia-19/publication/262921640_An_Empirical_Comparison_of_Botnet_Detection_Methods/links/6015ad13299bf1b33e38cb74/An-Empirical-Comparison-of-Botnet-Detection-Methods.pdf : original author PDF accessible through web tool, despite direct HTTP403; dataset and ground-truth sections and manuscript metadata. |
| moustafa2021ton | https://www.researchgate.net/publication/352055999_A_new_distributed_architecture_for_evaluating_AI-based_security_systems_at_the_edge_Network_TON_IoT_datasets : original full-text via web, direct HTTP403; edge/fog/cloud testbed and dataset collection sections, network/telemetry/OS sources. |
| xiang2026can | https://www.preprints.org/manuscript/202609.1557/v1 : original full-text via web, direct HTTP403; normal-only adaptation/calibration separation, empirical block ranks and maxima, block-length dependence/resolution constraints, unresolved exchangeability and explicit preprint status. |
| ha2026targetfpr | https://mdpi-res.com/d_attachment/applsci/applsci-16-02284/article_deploy/applsci-16-02284.pdf : benign validation target-FPR threshold transferred unchanged, labeled seen attacks, host-session graphs and realized violation reporting. |
| nguyen2025fedmse | https://www.jaist.ac.jp/~razvan/publications/fedmse_iot_network_intrusion_detection.pdf : SAE-CEN benign representation/centroid scoring, processed development set, MSEAvg inverse-error weighting and model-parameter aggregation. |


