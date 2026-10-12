# PACE manuscript verification

Audit date: 12 October 2026. Status: READY FOR AUTHOR REVIEW. Preparation, verification and packaging are complete. Author declarations and human approval remain required before submission.

## Scope, evidence and integrity

The governing request is the attached pasted-text-1.txt (attachment 571cf1f6-867b-4b48-9247-5017172d363a). Persistent changes are confined to manuscript/. No model training, campaign execution, report regeneration, implementation changes, commits or pushes were performed. Read-only recomputation of summary statistics from frozen outcomes is distinguished from new experiments.

Source hierarchy: executed implementation/configuration; verified frozen outputs and numerical tables; reporting code; documentation; narrative. Read README, both docs/PACE.md and docs/TECHNICAL_ARCHITECTURE.md, configs/default.yaml, relevant core/experiment/dataset/reporting/types modules and behavior tests. No applicable AGENTS.md was found. Result inventory, corrected from early estimates: 15 CSV tables with paired TeX files, 17 PNG figures with PDF counterparts. All original figures were screened on a contact sheet; selected figures were inspected individually. No prior author publication was supplied; absence of undisclosed self-overlap cannot be certified.

Reviewer B independently checked all 110 campaign and 110 report artifacts: schema/content digests and current implementation hashes agree; report source-artifact references correspond. Campaign implementation SHA256: 7dd88ce13b0d32a3875db47869e5ac13f5cecfa0dc82c6d14322dc67419a8a3b. Reporting implementation: d294c7fbf6f1fbff4899ea12c39351be26fc4c4d4824e77734330e4b79e505ff. Read-only Python used PYTHONDONTWRITEBYTECODE=1. This is not a reconstruction of raw datasets or preprocessing digests. Runtime contributes to the provenance digest; artifacts do not contain the separately asserted environment object found in earlier documentation.

Initial tracked worktree was clean. Subsequent git status shows only untracked manuscript/. The three included figure PDFs were SHA256-identical to their results/figures originals. Final git status confirms only manuscript/ is untracked; git diff is empty, so no tracked scientific file differs from HEAD.

## Scientific traceability and resolved discrepancies

| Manuscript content | Authoritative evidence and verified interpretation |
|---|---|
| Evidence transformation | core/evidence.py: upper tail includes ties, smoothed (count+1)/(n+1), negative log; beyond-reference extrapolation is an additional log1p term using max-minus-median with 1e-6 floor. It is not a calibrated p-value. |
| Block thresholds | core/decision.py: J=max(1,M//250), balanced consecutive np.array_split blocks, zero-based higher quantile ceil((1-w)(n_j-1)), maximum across block quantiles; alert uses strict >. Not temporal block-maximum rank testing. |
| Union scaling | Same module: starts at multiplier 1, doubles up to 64/cap, then 20 bisections. Singleton anchor also passes through union calibration. A calibration constraint is not a population guarantee. |
| Low-alpha allocation | _anchor_tier_budget uses configured 1/250. With alpha=.01, reserve=anchor fraction=.5, allocations are peer .005, anchor .005, bank .0025; sum .0125. There is no final aggregate recalibration. At .05, sum is .05, still without a test-time guarantee. |
| Local models | experiment/models.py: six-score anchor excludes AE; primary local mean comparator has seven scores including AE. Feature bank reuses IF/HBOS/Gaussian outputs and adds marginal/pair rarity plus response IF. No deterministic non-inferiority or isolated AE-removal effect follows. |
| Rarity details | 150 ordered feature pairs sampled with replacement; self-pairs dropped for d>1, duplicates retained. Unique internal eighth-quantile cuts, left insertion, at most eight active codes but 81 smoothing cells. Marginal score differs from original ECOD; explicitly called ECOD-inspired. |
| Fitting and information | runner.prepare_clients uses all clients' full benign training roles for pooled scaling, including target rows beyond N. R/C are additional. Feature-space IF/LOF/kNN fitting cap is 8000; response IF fits N response vectors. Shared benchmark still trains AE. No measured computation or communication claim. |
| Targets and sampling | types.py manifests precede eligibility. Adapters/common code establish 58 targets, six corpora/seven strata, dimensions 36/29/37/38/115/15/13. Targets include scenarios and hosts. Retained-label caps, thinning, deduplication and role-specific chronological splitting are disclosed; no global wall-clock holdout. |
| Gotham dependency | Adapter calculates five whole-second features on all packets before labels/reservoir sampling. Benign rows can contain attack/unlabeled contemporaneous information; later packets affect earlier rows and role boundaries can share raw contributors. This is not a claim of direct attack-label fitting. |
| Statistical estimator | reporting/metrics.py: attack types and three seeds averaged before target effects; 10,000 target bootstrap draws within DatasetName, seed314159, linear percentile interval. UNSW groups share one corpus stratum. Shared peers/captures and development selection are not modeled as clusters. |
| Matched operating points | Primary per-seed sorted-FPR np.interp clamps endpoints and does not enforce monotonicity. Distinct max-fusion routine uses an envelope/coverage exclusion. Primary outside-coverage counts /174: 75,69,36,28 in alpha,N order below. “Clamped sweep-adjusted” is the supported description. |
| Acceptance | reporting/acceptance.py checks severe/harmed target counts and relative FPR allowances. It is an engineering rule, not statistical non-inferiority or a test of FPR<=alpha. No preregistration established. |

### Numerical claims

Order throughout: (.01,100), (.01,400), (.05,100), (.05,400). Values are target-macro quantities, not traffic-weighted operational rates.

- headline.csv: nominal gains .026086/.028306/.023418/.023219; clamped sweep-adjusted gains .024659/.025735/.022176/.020718. Reviewer B independently reproduced raw nominal means and all four adjusted means/intervals from frozen campaign outcomes.
- Excluding TON_IoT gains: .009306/.010228/.005536/.008299. strata.csv TON gains: .117448/.126729/.120780/.104447. Several other-stratum intervals include zero; N-BaIoT gains are negligible.
- false-alert-rates.csv: PACE mean .033992/.032369/.055453/.054337; comparator .030516/.030781/.057437/.064407. Median .006485/.006419/.019118/.018427; p90 .062058/.067903/.162115/.159692. Distribution is over seed-mean target rates. Worst IoT-23 7-1 rate is .669005/.670842 at .01, .671844 at .05.
- targets.csv: CTU-13 scenario12 loses .032595 at .01/400. Primary engineering severe/harmed counts do not prove population non-inferiority.
- scaling.csv and budget-effects.csv: 47-target N2000 matched intervals include zero; .05 excluding TON loses .000370, interval [-.000712,-.000074]. At N20000/15 targets nominal gains .005329/-.001311 have intervals including zero. Within that cohort .05 detection drops .028296 from N100 to N20000, interval [-.032827,-.023766].
- seeds.csv: seed6 excluded-TON gain at .05/400 is -.000195. Ablation, decomposition and sensitivity tables support only the bounded comparisons stated in Results; removals change calibration allocations and cannot yield additive causal attribution. The response-specific contribution is small.
- Native manuscript tables transcribe unchanged result values; three existing vector figures show false-alert distribution, target gains and within-cohort scaling. No new scientific figure/data generation.

## Publisher preparation

JISA is the primary target; Internet of Things and JNCA are alternatives. Official ScienceDirect guides and scope pages returned403, so no journal-specific word limit, mandatory graphical abstract/highlights rule, or reference-style requirement is claimed verified.

Verified Elsevier-wide sources:
- [LaTeX instructions](https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions): elsarticle and source submission; eventual Editorial Manager source package may require a flat folder. Working source remains modular.
- [Artwork](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-overview) and [sizing](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing): PDF vector artwork supported. Existing vector figures are retained.
- [Generative-AI policy](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals): substantive assistance requires disclosure and human responsibility. Draft names Codex and its actual uses without falsely asserting completed human approval.

Readable single-column elsarticle preprint12pt, numbered equations, native booktabs tables, numerical BibTeX style, approximately220-word abstract, six keywords and four highlights are editorial choices consistent with verified general guidance. No generated imagery used.

## DOI and original-content verification

Metadata authority is Crossref for publications and DataCite for the two Zenodo datasets, cross-checked with original/publisher records. Exact verified author lists, titles, years, venues, volumes/issues and pages/article numbers are stored in references.bib; table keys map uniquely to those records. Every admitted DOI was separately resolved to its intended publisher/record, generally HTTP302. Full-text access was checked independently. “Inspected” means the specified original content was read, not every page or proof. Published metadata takes precedence over preprint titles/dates. DOI presence alone was never used as evidence of a method.

| Key; DOI | Accessible original source and inspected content | Supported manuscript use |
|---|---|---|
| yue2026mcde; 10.3390/e28091026 | [EuropePMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13605448/fullTextXML), sections3 and4, algorithms, transfer/limitations | Closest calibrated supervised/benign-mode fusion; separate final threshold, family holdout; no PACE superiority claim. |
| barber2023nonexchangeable; 10.1214/23-AOS2276 | [Author preprint](https://arxiv.org/pdf/2202.13415), pp1–5,10–12/3.1 | Weighted quantiles and coverage-gap assumptions; not implemented by PACE. |
| bates2023outliers; 10.1214/22-AOS2244 | [Author preprint](https://arxiv.org/pdf/2104.08279), pp2–4 definitions/dependence | Marginal versus conditional validity; shared calibration dependence. |
| angelopoulos2023conformal; 10.1561/2200000101 | [Author preprint](https://arxiv.org/pdf/2107.07511), pp4–5 procedure | Split-conformal correction and marginal coverage; journal title differs from preprint. |
| liu2008isolation; 10.1109/ICDM.2008.17 | [Author PDF](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf), sections2–4, algorithms1–3 | Credit random-partition/path-length estimator. |
| breunig2000lof; 10.1145/342009.335388 | [Publisher PDF](https://sigmodrecord.org/?download_id=6631&smd_process_download=1), sections3–4 definitions3–7 | Credit relative-density estimator. |
| pang2021review; 10.1145/3439950 | [Author preprint](https://arxiv.org/pdf/2007.02500), sections4–5.1.1 | Reconstruction objective limitations, not an explanation of measured PACE gains. |
| diot2019; 10.1109/ICDCS.2019.00080 | [Author preprint](https://arxiv.org/pdf/1804.07474), II–IV | Device-type GRU and sample-weighted iterative federation; PACE is not FL. |
| kitsune2018; 10.14722/ndss.2018.23204 | [Author preprint](https://arxiv.org/pdf/1802.09089), IV-D algorithms4–5,V | Prior anomaly modeling in detector-error space. |
| nbaiot2018; 10.1109/MPRV.2018.03367731 | [Author preprint](https://arxiv.org/pdf/1805.03409), III–IV | Benign-only device-local AE, corpus credit; published author order used. |
| osdn2024; 10.1145/3639822 | [Author preprint](https://arxiv.org/pdf/2311.11249),3–5,eq15 | Labeled source/unlabeled mixed target, heterogeneous alignment. Crossref pp1–26 retained; article ID not inferred from secondary source. |
| talpini2025; 10.5220/0013376600003890 | [Author PDF](https://marcosavi.altervista.org/Papers/2025_ICAART_Federated_Calibration_IDS.pdf),3–5/algorithm1 | Labeled Platt/FedAvg probability calibration differs from benign alert thresholds. |
| vim2022; 10.1109/CVPR52688.2022.00487 | [Author preprint](https://arxiv.org/pdf/2203.10807),3–4,eq4–8,6 | Complementary logit/residual OOD evidence; not a direct IDS baseline. |
| zimek2014; 10.1145/2594473.2594476 | [Publisher PDF](https://www.kdd.org/exploration_files/V15-01-02-Zimek.pdf),4.2,5.1–5.2 | Normalization/diversity/max domination precedent. Verified2014,15(1),11–22; stray33974 field removed. |
| hyperids2025; 10.1038/s41598-025-11659-7 | [EuropePMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12485126/fullTextXML), optimization/client training/evaluation | Labeled hypernetwork personalization differs from frozen peers. Article33974 independently verified. |
| ecod2022; 10.1109/TKDE.2022.3159580 | [Author preprint](https://arxiv.org/pdf/2201.00382),3.2–3.3/algorithm1 | Original max of three aggregate tails differs from implemented featurewise-max sum. Journal year2023. |
| jin2026crossprotocol; 10.3390/s26041184 | [EuropePMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12944665/fullTextXML),3.1–3.5,4.6–4.7,discussion | Capture/split and transferred-threshold sensitivity; no imported performance. |
| garcia2020iot23; 10.5281/zenodo.4743746 | [Original dataset description](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/), scenario table | Dataset record, not scholarly full-text publication;20 malicious/3 benign source captures. |
| belarbi2025gothamdata; 10.5281/zenodo.14502760 | [Original Zenodo documentation](https://zenodo.org/records/14502760), description/metadata | Dataset record,78 emulated devices; registered title differs from preprint; PACE subset separate. |
| vovk2020averaging; 10.1093/biomet/asaa027 | [Author PDF](https://sas.uwaterloo.ca/~wang/papers/2019Vovk-Wang-BIOM.pdf),1–3 | Valid p-inputs required for formal combination; empirical thresholds supply no population premise. |
| sommer2010closed; 10.1109/SP.2010.25 | [Author PDF](https://www.icir.org/robin/papers/oakland10-ml.pdf),III-B/C,IV-B/C/D | Semantic gap and operational evaluation; not2026 adoption statistics. |
| saito2015precision; 10.1371/journal.pone.0118432 | [Publisher full text](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432), metrics/PR baseline/simulation | Precision depends on prevalence; no operational precision derived from macro rates. |
| arp2024pitfalls; 10.1145/3643456 | [Author-hosted publisher PDF](https://www.mlsec.tu-berlin.de/docs/2024-cacm.pdf),P1–P10,3–4 | Selection, baseline and lab-evaluation limitations; DOI-bearing CACM version, not double-counted USENIX work. |
| hamza2019mud; 10.1145/3314148.3314352 | [Author PDF](https://www.contrib.andrew.cmu.edu/~theophib/papers/SoSR19.pdf),3.1–3.3; [official data](https://iotanalytics.unsw.edu.au/attack-data.html) | UNSW minute MUD-flow data provenance; not UNSW-NB15. |
| garcia2014ctu; 10.1016/j.cose.2014.05.011 | [Author-uploaded PDF](https://www.researchgate.net/profile/Sebastian-Garcia-19/publication/262921640_An_Empirical_Comparison_of_Botnet_Detection_Methods/links/6015ad13299bf1b33e38cb74/An-Empirical-Comparison-of-Botnet-Detection-Methods.pdf),6/8.1 | CTU-13 creation/labels; original split is not attributed to PACE. |
| moustafa2021ton; 10.1016/j.scs.2021.102994 | [Author full text](https://www.researchgate.net/publication/352055999_A_new_distributed_architecture_for_evaluating_AI-based_security_systems_at_the_edge_Network_TON_IoT_datasets),3–4,feature descriptions/start6.2 | Corpus architecture/heterogeneity; no original IP-model feature claim attributed to PACE. |
| xiang2026can; 10.20944/preprints202609.1557.v1 | [Original full text](https://www.preprints.org/manuscript/202609.1557/v1),2.1–2.3,3.5,4 | Normal-only adaptation and block-maximum ranks; different decision unit. Non-peer-reviewed preprint explicitly labeled; posted18September2026. |
| ha2026targetfpr; 10.3390/app16052284 | [Publisher PDF](https://mdpi-res.com/d_attachment/applsci/applsci-16-02284/article_deploy/applsci-16-02284.pdf),3.1–3.5,4.1–4.2,limitations,eq2–5 | Benign-validation target-FPR threshold transfer, empirical tail-risk reporting; supervised host-session graphs, not target-benign peer unions. |

| nguyen2025fedmse; 10.1016/j.cose.2025.104337 | [Author accepted manuscript](https://www.jaist.ac.jp/~razvan/publications/fedmse_iot_network_intrusion_detection.pdf),pp4–9,4.1–4.2,algorithms1–2,eq4–6,5.1 | Benign-only AE/centroid and inverse-MSE aggregation; shared processed benign server set. Metadata: Nguyen and Beuran, Computers & Security151(2025)104337. DOI302 to S0167404825000264. |

Search is focused, not systematic or proof of uniqueness. Currently29 references:27 scholarly publications (one preprint) and2 datasets; original content inspected for all. The40–60 reference suggestion is a quality target, not a reason to admit inaccessible or redundant citations.

Excluded from method-level synthesis after access attempts: CROSS-IoT10.1002/dac.70565; multi-source adaptation10.1016/j.eswa.2024.124352; cross-network transfer10.1016/j.comnet.2026.112434; Choir-IDS10.1016/j.inffus.2025.103473; Shiomoto NOMS10.1109/NOMS69089.2026.11668140. Metadata/abstracts or previews were available, sufficient original methods were not. Shiomoto's abstract alpha denotes Dirichlet heterogeneity, not an FPR budget. Krstajic10.1186/1758-2946-6-10 was read/verified but omitted as redundant with the relevant Arp discussion. FedMSE was admitted after independent original-method and DOI checks. DA-DCC-IDS10.1109/ACCESS.2026.3732976 metadata and DOI redirect were verified (IEEE Access14,150978–151004); institutional abstract was accessible but IEEE full text was not, so excluded from substantive comparison. No SciSpace service/page was used; incidental search hits were ignored.

## Independent hostile reviews and resolution

Three independent agents completed initial investigation and repeat review after revisions. Temporary service limits delayed but did not replace these reviews.

| Reviewer | Valid criticism/classification | Correction and repeat-review disposition |
|---|---|---|
| A: novelty | Unsupported novelty and incomplete precedent: maximum domination, response-space detection, empirical rarity and calibrated hybrid fusion already exist. | Zimek/Kitsune/ECOD/MCDE credited; contribution limited to specified integration and offline evaluation. OSDN/Talpini/FL information regimes distinguished. Final reread found no remaining major misrepresentation; metadata typo corrected. Integration-only novelty and missing external baselines remain scientific concerns that can justify rejection. |
| B: methods/statistics | Factual/presentation errors: allocation sum, anchor scaling, quantiles, matched-FPR clamping, dependence, engineering versus statistical non-inferiority. Missing analysis of existing outputs. | Exact rules and .0125 sum stated; raw summaries/hashes independently checked; clamping counts included; uncertainty and acceptance qualified. Repeat review found no new factual methodological error. Development reuse/dependent targets/limited seeds remain limitations. |
| C: deployment | Unsupported total-data/cold-start, privacy/distribution, physical-device and online-detection implications; missing adapter details. | Pooled full-training scaling, additional R/C, mixed target units, Gotham aggregation and absent latency/drift/adversarial evidence disclosed. Final minor manifest, feature-space cap and thinning details added. Reviewer C rechecked and confirmed these last three edits, including the TON intermediate reservoir bound. |

No reviewer verdict establishes journal acceptance, uniqueness, or real-world security. No new experiments were introduced to address criticisms.

## Author confirmations and final gates

Names/affiliations/correspondence; CRediT; funding/conflicts/acknowledgements; public code release and immutable identifier; dataset redistribution conditions; prior/overlapping publications; final human scientific review and AI-assistance responsibility require author confirmation. These are honestly pending in front matter/declarations.

Unresolved science: development-selected targets, shared-peer/capture dependencies, class-aware sampling and no prospective global holdout, Gotham aggregate lookahead, total target data beyond N, no final population FPR control, severe heterogeneous false alerts, concentrated gains, incomplete external baselines and absent malicious-peer/drift/resource/deployment evaluations. Writing can bound these findings but cannot cure them.

Build: cd manuscript && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex. Available ordinary tools: TeX Live2023, elsarticle3.3, latexmk/BibTeX/Poppler. Main-text count uses texcount -total sections/*.tex, excluding headings/captions/front matter/references. Final counts:8,365 main-text words,185 heading words,276 caption words;26 PDF pages. Abstract whitespace count208; six keywords and four highlights.

Final verification: clean-state build in a fresh output directory succeeded with latexmk/pdflatex/BibTeX. No LaTeX warnings, missing citations/references, overfull or underfull boxes. All26 final rendered pages inspected; figures, tables, equations and page breaks are legible and unclipped. PDF SHA256:4c89da3f106f111a1ca31c27ea4f37e6cc6a1e0c31352db48f03d3ed6c1eacbe.

Bibliographic consistency:29 unique keys and29 unique DOIs; no missing or uncited entries.27 included scholarly publications with original content inspected, plus two datasets with original documentation. The additional Krstajic publication was inspected but omitted. Included figure PDFs remain byte-identical to originals.

Final Unicode inspection found zero suspicious hits across main.tex and all nine sections. An18-word normalized exact-overlap check found zero matches with either repository documentation file. This limited check is not a plagiarism certificate; no prior author publication was supplied.

Scope wording follows the author's preference: conclusions are limited to the58-target/six-corpus benchmark. Provenance remains explicit in this audit: docs/PACE.md:111 describes earlier outcome-informed design choices,125 states identical targets across development phases,174 distinguishes untouched seeds from targets. No independent target validation is asserted.

All nine requested manuscript sections, DOI/full-text verification, closest-work comparison, three hostile reviews/reviews after revision, implementation/statistical corrections, adverse results and verified Elsevier-wide preparation requirements are addressed. Current journal-specific guides remain inaccessible; author confirmation is required. Scientific limitations remain substantive and can affect journal suitability. The internal engineering acceptance result does not establish journal readiness.

Packaging cleanup is complete. Initial broad cleanup commands were rejected by automatic approval review; explicit literal-file removal succeeded after inventory verification, followed by removal of the verified-empty directories. Only permanent manuscript deliverables remain.

Clean-build verification was repeated after removing every auxiliary and bibliography intermediate. BibTeX regenerated main.bbl successfully;26 pages, no relevant warnings, and pdftotext -layout output identical to the visually inspected delivered PDF. The delivered PDF hash above is unchanged.

Final editorial assessment: READY FOR AUTHOR REVIEW. Three independent hostile review passes and repeat reviews identified and corrected manuscript errors. Remaining scientific limitations are stated within the benchmark scope; this assessment is not journal acceptance, a deployment guarantee, or permission to omit author confirmation. The final package contains the complete source, compiled PDF,29-entry DOI bibliography, three original vector figures, four native tables, highlights and this consolidated audit.