# PACE final revision report

Revision date: 22 October 2026. This report describes the revised master, not an acceptance guarantee. No algorithm, implementation, configuration, test, dependency, dataset or frozen result was changed. No model was trained or rescored. No commit or push was made.

## Outcome and scope

The master now uses generic `article` LaTeX with a numbered bibliography, 11-point type and 25 mm margins. It has 25 pages through the conclusion and 31 pages overall. Main prose is 8,551 TeXcount words, within the requested 8,000–9,500 target. All primary results, adverse outcomes, four figures, the complete comparison inventory and dataset provenance remain accessible in one PDF.

The strongest supported conclusion is an architectural result on a reused benchmark: replacing the complete mean-local branch improves detection in four primary settings, with lower mean FPR in both alpha = 0.05 settings. Separate-peer versus fused-peer decisions are a different comparison and incur higher FPR. Local-exclusive coverage describes complementary detections, not comparator gains. Neither the paper nor this report claims general equal-FPR superiority, a calibration theorem, uniform target benefit or deployment readiness.

## Scientific and methodological changes

The abstract and introduction now organize the contribution into three findings: local-branch replacement with a common peer branch; peer-wise versus fused decisions with the complete local branch fixed; and complementary detection. The final introduction explicitly distinguishes family ablation effects on final detection from local-exclusive coverage. Peer drowning remains a motivation rather than a demonstrated causal mechanism.

Implementation inspection confirmed frozen supervised peers; attack-minus-benign margin evidence; tie-inclusive target-relative empirical tails and extrapolation; peer-specific thresholds; blockwise empirical quantiles; shared peer-union scaling; the six-detector anchor without the comparator's autoencoder; the six-expert bank; separate calibration tiers; and final OR without aggregate recalibration. The low-alpha resolution exception is retained: the implemented allocations need not sum to nominal alpha. No mathematical guarantee was added. Equations and implementation conventions were preserved.

The main evaluation protocol remains explicit about attack-type weighting, three learning seeds, shared peers, target eligibility, pooled preprocessing and the extra reference/calibration data beyond fitting budget N. The lengthy adapter inventory moved from the main experimental section into Appendix B. Gotham's complete-second aggregates and cross-role contributors remain in the main information-regime discussion. Available history does not establish that every protocol/arm decision preceded outcome inspection; draft requests for confirmation were replaced by that factual boundary.

Results now emphasize joint detection/FPR improvement at alpha = 0.05 and the FPR cost at alpha = 0.01. The reserve sensitivity receives an explicit interpretation: reserve 0.25 lowers detection by 1.94–2.01 percentage points at alpha = 0.01, but raises detection by 0.23–0.50 points and FPR by 0.46–0.60 points at alpha = 0.05. The default is not uniformly optimal. The anchor-free and single-union alternatives remain fully reported, including equal aggregate low-alpha detection for the single-union arm.

## Statistical and frozen-evidence verification

The read-only reporting verifier validated 110 canonical JSON digests, 1,824 records, 408 identical repeats and 1,416 unique target/seed/budget/alpha keys. There are 21,240 method-rate records across 15 rules. Repeated keys were checked for equality rather than treated as additional observations. The primary inventory contains 60 rows: 15 rules at four settings.

Target-macro detection averages attack types equally within target/seed, then seeds 6–8, then targets. Bootstrap verification used 10,000 paired within-corpus resamples, NumPy seed 314159, the implementation's corpus and lexical target ordering, target-count weighting and linear percentile endpoints. The two UNSW strata form one corpus. These conditional intervals do not resample peers, attacks, seeds or new deployment populations.

| Nominal alpha | N | Detection gain, percentage points | 95% paired interval, percentage points | Mean FPR change, percentage points |
|---|---:|---:|---|---:|
| 0.01 | 100 | 2.6086 | [1.1323, 4.3346] | +0.3476 |
| 0.01 | 400 | 2.8306 | [1.2339, 4.7375] | +0.1588 |
| 0.05 | 100 | 2.3418 | [0.5777, 4.3968] | -0.1984 |
| 0.05 | 400 | 2.3219 | [0.3116, 4.5968] | -1.0070 |

The independent experimental-rigor reviewer reproduced all primary detection/FPR differences and intervals, the repeated-key handling, and the new common-support analysis directly from frozen records. Excluding TON_IoT leaves gains of 0.55–1.02 percentage points; their small lower endpoints and outcome-selected deletion analyses remain qualified. Win/tie/loss counts, the CTU-13 scenario-12 regression, fixed-cohort budgets and the high-resource TON_IoT host loss remain reported. Larger N does not establish monotonic benefit. Engineering acceptance remains distinct from statistical non-inferiority.

The worst-FPR target is IoT-23 capture 7-1, near 67% under both rules. It has 1,996 calibration observations. About 66% of its benign-test rows exceed the calibration range of `orig_ip_bytes`. The calibration-size association is explicitly confined to 58 seed-averaged targets at alpha = 0.01, N = 100; corpus effects are confounded. Frozen results lack the per-expert extrapolation flags needed to establish a causal diagnosis. A material PACE-specific regression is now explicit: capture 34-1 has 25.54% FPR versus the comparator's 11.52% at alpha = 0.01, N = 100. Similar worst-target rates therefore do not imply identical target-wise costs.

## Operating-point analysis

`verify_common_support.py` adds one reproducible, read-only descriptive analysis. It retains the reporting routine's stable FPR sorting, endpoint precedence and interpolation. It does not retrain, score new observations, extrapolate or select the most favorable tied sweep point.

| alpha / N | Covered cells / 174 | Targets covered in all three seeds | Supported-subset gain, percentage points | 95% interval |
|---|---:|---:|---:|---|
| .01 / 100 | 99 | 20 | 1.100 | [0.005, 2.419] |
| .01 / 400 | 105 | 26 | -0.009 | [-0.128, 0.114] |
| .05 / 100 | 138 | 36 | 3.344 | [0.753, 6.103] |
| .05 / 400 | 146 | 41 | 1.731 | [0.076, 4.068] |

Below/above-support counts are 47/28, 49/20, 12/24 and 9/19. Exact equality to a stored empirical FPR occurs in 72, 59, 35 and 41 cells; 3, 6, 1 and 2 cells have tied matching FPRs with different TPRs. These facts restrict interpretation even within support. Subsets are outcome-dependent and differ across settings. Their intervals describe those selected subsets; they neither replace the full cohort nor estimate excluded targets. Test-FPR adjustment is retrospective. The earlier clamped gains remain reported as approximations, and neither calculation resolves equal-FPR superiority over the fixed-local peer-fusion arms.

## References and source support

The bibliography contains 38 entries, up from 37. The added article is Livinus Obiora Nweke, *Tail-calibrated mixture of experts for ultra-low false-alarm intrusion detection in IEC 61850 communications*, Discover Computing 29, article 434 (2026), [DOI 10.1007/s10791-026-10343-2](https://link.springer.com/article/10.1007/s10791-026-10343-2). Publisher methods and fresh Crossref metadata support the new comparison. Its two autoencoders, timing detector, tail evidence and empirical final fusion threshold are close precedent; it does not reuse foreign supervised peers. No performance ranking is implied.

All existing bibliographic identities were compared field by field with primary registry records and available publisher/source pages: title, authors, venue, year, volume/issue when applicable, pages/article identifiers and DOI. The fresh registry pass returned 26 of the original 37 records; 11 requests were rate-limited, so their already archived primary records were cross-checked rather than falsely reported as newly fetched. Fresh response bodies and failures are preserved in `final-bibliography-refresh.json`; the new article has a separate metadata record. The source registry's malformed UCI creator fields were not copied: six identifiable creators are retained from the [UCI dataset page](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot), with its duplicate/incomplete Asaf entry documented. ECOD remains the 2023 journal article despite its 2022 DOI/key; article numbering and accented names were preserved. Xiang remains explicitly labeled a non-peer-reviewed version-1 preprint.

Targeted passage checks cover MCDE's family holdout and device-transfer recalibration, Nweke's empirical Fisher-score threshold, the conformal validity distinctions, threshold-allocation precedent, federated parameter aggregation versus frozen peers, response-space detection, and recent operating-point studies. The novelty reviewer independently checked MCDE, Nweke, Zimek, Bridges, Angelopoulos/Bates, Bates et al., Vovk/Wang, Barber et al., FedMSE, the hypernetwork IDS, Talpini, DI-NIDS, N-BaIoT, Rey and ViM. The primary reviewer also checked accessible Xiang, Jin, Ha and Nouman passages. Not every publisher PDF was accessible; metadata verification is distinguished from reading an entire source. The LOF institutional link now redirects to a laboratory homepage, so no fresh full-text-access claim is made for it. No detailed new claim depends on inaccessible text.

Available local reference texts were also screened for long exact phrase overlap. That bounded comparison is not a plagiarism clearance, a comprehensive database search or proof of human authorship. It does not replace assessment of attribution and scientific originality. No AI-detection claim is made.

## Compression and supporting material

Counts use `scripts/count_manuscript_words.py`, which invokes TeXcount and excludes headings, mathematical expressions, citation commands, float/table bodies and bibliography from main prose. Captions and appendix prose are counted separately. Table contents are inventoried rather than mixed into a word count dominated by numeric cells. The abstract contains 222 counted prose words; a word processor may count mathematical tokens differently.

| Category | Before | After |
|---|---:|---:|
| Main prose, sections 1–9 | 9,386 | 8,551 |
| Abstract prose | 218 | 222 |
| Appendix prose, excluding tables | 221 | 377 |
| Captions | 424 | 474 |
| Bibliographic entries | 37 | 38 |
| Main article pages through conclusion | about 28 | 25 |
| Complete PDF pages | 35 | 31 |

Main prose fell by 835 words (8.9%). Introduction fell from 569 to 377 words, related work from 1,513 to 1,207, experimental methods from 2,040 to 1,880, discussion from 710 to 576, and conclusion from 197 to 142. Results grew from 2,240 to 2,252 because support coverage, reserve sensitivity and target-specific FPR costs deserved explicit reporting. Method equations and essential definitions were retained. Page savings reflect both editing and the change from 12-point Elsevier formatting to an 11-point generic article; they are not attributed entirely to prose compression.

No standalone supplement was created. The full 60-row comparison inventory and dataset provenance remain attached as Appendices A and B. Only the adapter preprocessing/sampling inventory moved from main text to Appendix B. This preserves access while the journal is undecided. An eventual journal may accept the appendices as supplementary files; they must then remain cited and supplied. Four figures remain necessary: architecture, target gains, FPR distributions and fixed-cohort budget effects. A new main-text table makes sweep coverage and selected-subset uncertainty visible.

## Official journal requirements and strategy

Official sources were checked on 21 October 2026; alternate publisher-link and IEEE policy checks continued on 22 October 2026. “No limit found” means the accessed guide did not state one; it is not a guarantee of unlimited length. “Unverified” means the official page could not be read reliably. Third-party summaries were not used to fill gaps.

### Journal of Information Security and Applications

The [official author guide](https://www.sciencedirect.com/journal/journal-of-information-security-and-applications/publish/guide-for-authors) was read in the browser. Security research fits its scope, but it explicitly screens out routine applications/combinations of established ML methods on standard datasets without a substantial security advance. This is a real editorial risk for PACE. Research-article preparation: abstract at most 250 words; 1–7 keywords; encouraged highlights (3–5, at most 85 characters each) and graphical abstract. No whole-article word/page or figure/table/reference-count cap was found. Editable LaTeX sources are accepted; PDF alone is insufficient. Numbered sections, consistent initial references, separate acknowledgements, contribution roles, interests and funding disclosures apply. The guide requests short author biographies and photographs. Supplementary files are cited, captioned and published as supplied. No page-triggered fee was found in the accessed guide; open-access charges are separate and should be checked after selection.

### Internet of Things

The [official Elsevier scope](https://shop.elsevier.com/journals/internet-of-things/2542-6605) covers IoT/AIoT security, reliability and privacy and includes full research papers. This is a plausible first candidate on subject fit. The [author-guide URL](https://www.sciencedirect.com/journal/internet-of-things/publish/guide-for-authors) repeatedly returned HTTP 403; the local browser subsequently became disabled. Consequently article word/page limits, abstract limits, figure/table/reference limits, supplementary rules, precise structure, journal-specific declarations, initial formatting and any page charges remain **unverified**, not assumed identical to JISA. Verify those fields before submission. Publisher-wide AI policy below is independently accessible.

### Computer Networks

The [official Elsevier scope](https://shop.elsevier.com/journals/computer-networks/1389-1286) includes network security and intrusion detection and seeks original complete research contributions. PACE fits that topic, but its networking/system implications need a clear cover-letter argument. The [official guide](https://www.sciencedirect.com/journal/computer-networks/publish/guide-for-authors) also returned HTTP 403. Word/page limits, abstract limits, figure/table/reference restrictions, supplementary rules, precise structure, journal-specific declarations, initial format and page charges remain **unverified**. In particular, no third-party claim of a 12- or 20-page limit was adopted.

### Cybersecurity

The [Research instructions](https://link.springer.com/journal/42400/submission-guidelines/research) support a research-article route: 150–250-word abstract, 3–10 keywords, concise section headings and an abbreviation list. Required declarations include data/material availability, interests, funding, author contributions and acknowledgements; irrelevant sections require an explicit not-applicable statement. Thus this journal would require adaptation of the user's declaration-free master. No article word/page or figure/table/reference-count cap was found. The [general guide](https://link.springer.com/journal/42400/submission-guidelines) requests double spacing, line/page numbers, no forced page breaks and editable sources; its LaTeX template is encouraged. Figure titles are limited to 15 words and legends to 300; individual figure files are limited to 10 MB and additional files to 20 MB. Tables remain editable. Additional files are published as supplied. No page-triggered charge was found in these guides; open-access APCs are a separate selection-stage check.

### IEEE Internet of Things Journal

The [official guide](https://ieee-iotj.org/guidelines-for-authors/) covers original substantial IoT systems and applications research. It requires the IEEE two-column template, a single-paragraph 150–250-word abstract without citations/display equations, and ORCIDs for all authors. Copyright and dissemination responsibilities apply. No hard article word/page cap or numerical figure/table/reference cap is stated on that page. The first eight published pages are a **charge threshold**, not a maximum: each additional page costs USD 175. The page lists optional open access at USD 2,695, with overlength charges separate; confirm the rate at submission. A graphical abstract is allowed and reviewed. The [IEEE supplementary-material guide](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/prepare-supplementary-materials/) requests separately labeled files during submission and recommends TXT/DOC/DOCX/PDF for supplementary text; datasets should include a README covering contents, environment, setup, use and outputs. The [IEEE AI policy](https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/) requires acknowledgement of generated content, naming the system, affected sections and extent of assistance; editing-only disclosure is recommended. Converting this master would change pagination; no two-column page estimate or fee quote is asserted. This is a higher-risk option without stronger systems validation.

### Computers & Security and recommendation

The [official Elsevier journal page](https://shop.elsevier.com/journals/computers-and-security/0167-4048) states an early-2024 moratorium on submissions with substantial AI/ML components, including security applications and security of AI/ML. PACE is not a viable recommendation under that published policy. Recently appearing ML papers do not override the stated submission restriction.

Prefer Internet of Things provisionally for scope, subject to completing its guide checks; Cybersecurity is a reasonable alternative after required adaptation. Consider JISA only with a strong argument for the specific security contribution, and Computer Networks if the network-level relevance is persuasive. This is an editorial judgment, not an acceptance forecast. No journal has been selected or contacted.

## Authorship, declarations and submission-only facts

The exact requested author order, affiliations, three email addresses, corresponding author and ORCID are inserted. The user confirms that the authors approved the paper, that there was no funding and no competing interest, and instructs that these subjects and individual roles not appear in the paper. The neutral master follows that instruction. It contains no draft-for-review or approval-pending statement. Individual contribution roles were not invented.

The public code URL is https://github.com/naslouby-salahe/pace. Public HEAD `41b3a18c10422fc41d1785c35881ccd6323941fa` was verified; protected implementation/results at the local starting revision match that public revision. The availability section promises public implementation/configurations/result tables, not that every locally available campaign JSON is already public. Detailed reporting reproduction additionally requires the inventoried campaign JSONs and cache manifests. A manuscript evidence inventory cannot substitute for their distribution.

Remaining author-only facts, only if demanded by the selected submission: individual contribution roles; the other two authors' ORCIDs for IEEE; biographies/photos for JISA; any missing dataset retrieval/version and redistribution records; and private evidence of prospective design/criterion/arm timing. The manuscript does not assume those facts. Author approval, funding and interest status are no longer unknown. Submission portals may require the known declarations even though the neutral master omits them. No submission was made.

The [Elsevier generative-AI policy](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals) requires disclosure for substantive assistance beyond basic spelling/grammar checks. The assistance here includes organization, wording, literature/source assistance and manuscript-only reporting scripts. A submission-specific statement, to adapt to the selected journal's required location, is:

> During preparation of this manuscript, the authors used OpenAI Codex to assist with manuscript organization and language, source checking, and the preparation and verification of read-only summaries of existing experimental artifacts. The authors reviewed and edited the assisted material and take responsibility for the content of the publication. The assistance did not retrain models or alter the frozen experimental campaign.

The user has confirmed author approval; this does not independently document review of each final edit. Authors should ensure that the responsibility statement accurately describes their final submission review. The journal-neutral master omits its former standalone draft disclosure as requested. Policy information remains here rather than being concealed or replaced with a false no-assistance declaration.

## Adversarial review and corrections

Four independent reviewer agents assessed the actual revision. Two initial attempts hit a service usage limit; both resumed successfully. These are bounded AI-assisted reviews, not external journal peer review.

| Reviewer | Main challenge | Resolution and remaining boundary |
|---|---|---|
| 1: scientific novelty | Calibration, score fusion and constituent detectors predate PACE; no external system was rerun. | MCDE/Nweke distinctions verified; architectural integration stated without comparative superiority. Optional Nweke table row was not necessary because its close comparison is explicit in prose. No substantive novelty misrepresentation found. |
| 2: experimental rigor | Result attribution, repeated keys, conditional bootstrap, operating-point support and reserve tradeoff. | Independent recomputation matched primary and support numbers. Reserve paragraph added and rechecked; no remaining numerical correction. External baselines and untouched validation remain open. |
| 3: IoT security | Trusted benign data, pooled scaling, Gotham dependencies and hidden target FPR costs. | Named worst target, reported calibration size/diagnostic setting and added capture-34-1 regression. Reviewer rechecked all corrections; no unsupported deployment claim remained. |
| 4: journal editor | Length, clarity, scope and distinction between coverage and ablation. | Finding C corrected to final detection; further discussion/conclusion compression removed the mostly empty final main-text page. Current contribution remains clear; journal adaptation is still needed. |

Reviewers 2 and 3 repeated their checks after corrections. The primary agent repeated the editorial/novelty assessment after final compression, preserving source distinctions, statistical qualifications and adverse findings. The editor's approximate presentation assessment was 9/10; the novelty reviewer described the contribution as moderate/incremental. An overall 9.5/10 scientific certification is not supported by this campaign. The manuscript is defensible for serious journal consideration as a bounded benchmark study; development reuse, absent comparable external baselines, operating-point fairness and extreme FPR remain material scientific limits.

## Optional future experiments, ranked

1. **Untouched target/capture validation:** highest impact on generalization and development-selection concerns; freeze architecture and decisions before inspection.
2. **Controlled equal-FPR evaluation and external baselines:** compare PACE, peer fusion and suitable MCDE-like alternatives under compatible information regimes, with calibration-selected operating points and sufficiently dense score support. Highest impact on superiority claims.
3. **Extreme-target FPR diagnosis:** inspect detector-level score/reference excursions and calibration/test changes on 7-1 and 34-1, then validate any correction on separate data. High practical impact.
4. **Prospective/capture-wise causal processing:** reconstruct features without cross-role/future contributions and evaluate untouched captures. High impact on streaming claims.
5. **Operational cost and incident evaluation:** measure latency, resources, aggregation and analyst-facing precision under realistic prevalence. Necessary for deployment claims.

None was executed. These are not blockers to the manuscript's current narrow architectural claims; they would be required for stronger generalization, superiority or deployment claims.

## Technical validation and files

The existing latexmk/BibTeX workflow builds successfully. Final checks cover citation/reference resolution, box warnings, metadata, draft markers, table agreement, Unicode hygiene, all rendered pages and protected-file hashes. Evidence is consolidated in `revision-evidence/final-validation.json`; current integrity data supersede historical logs. The PDF has 31 pages and contains no empty page or isolated conclusion page. All pages were rendered and inspected; charts retain full adverse tails and legends. No scientific test suite or campaign was run because protected executable behavior was not changed.

Actual modified/added paths are recorded in `revision-evidence/final-files.json`, generated from Git status. The changes comprise the master/PDF; introduction, related work, methodology float placement, experimental methods, results, discussion, conclusion and appendix sources; bibliography; prior-work float placement; audit/report/reproduction documentation; manuscript-only support/count scripts; and current verification evidence. Frozen scientific result tables and figure data are unchanged. The manuscript primary display table now expresses rates as percentages and differences as percentage points, checked against unrounded reporting evidence. No competing manuscript version was created in the repository; intermediate builds remained under `/tmp/pace-final` and `/tmp/pace-style-review`.

Two official journal author guides remain inaccessible/unverified even through the alternate official Elsevier links checked on 22 October. That limits submission-format certification, not the completed manuscript revision. The report explicitly records those gaps instead of inventing publisher requirements.

## Editorial follow-up, 22 October 2026

The subsequent user-requested review improved concrete wording, ablation interpretation, percentage/percentage-point presentation, float placement and page flow. The current totals above include this pass. Detailed findings and remaining scientific concerns are in `editorial_review.md`; verification is recorded in `revision-evidence/editorial-validation.json`.
