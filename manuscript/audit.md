# PACE manuscript audit

**Manuscript:** *PACE: Peer-Wise Calibrated Experts for IoT Intrusion Detection with Benign-Only Target Data*.
**Assessment date:** 20 October 2026. **Verified draft:** `manuscript/main.tex` and matching 28-page `manuscript/main.pdf`, PDF build date 13 October 2026.
**Project root:** `/home/naslouby/Projects/PACE`. Repository-relative evidence paths resolve under this root; section/page numbers refer to this unchanged draft.

**Scope and evidence standard:** independent read-only examination of the manuscript, relevant implementation, configuration, cache manifests/selected arrays, all 110 campaign artifacts, reporting calculations and primary literature. Only this canonical audit and the permitted reference PDF collection are updated. No training, experimental reruns, manuscript/code/results edits or history changes were made. Reproducing stored calculations is distinguished from independent campaign replication. All artifact content digests were checked; the current implementation digest matches. Raw-source checksums and all raw-capture duplicates were not exhaustively recomputed.

## 1. Executive Summary

**Overall score: 7.0 / 10.** The method is precisely specified, the primary numerical evidence reproduces and the manuscript already contains careful information-access and calibration caveats. The defensible novelty is the transfer-and-decision architecture, not a new anomaly estimator or FPR theorem. The principal weaknesses are incomplete arm reporting, limited four-point operating adjustment, development/cohort reuse and concentration of gains. Submitted-figure reproducibility and release declarations need completion. This is a quality judgment, not an acceptance probability.

The strongest supported results are:

- Replacing the shared-peer comparator’s complete mean-local branch raises target-macro detection by **2.61, 2.83, 2.34 and 2.32 percentage points**; all four conditional paired intervals exclude zero. Every primary learning seed has a positive pooled gain.
- At α=.05 both budgets also have **lower mean realized FPR**, an accurate aggregate result surviving removal of its two largest FPR drivers. Retain this claim (M19).
- With the local bank fixed, separate peer thresholds outperform mean/max fusion by **3.0–7.3 detection points at nominal settings**, with higher FPR. This is architectural evidence, not an isolated causal test of drowning or exact equal-FPR superiority (M5, M12).
- Rarity removal causes the largest nominal detection loss among tested expert families; local-exclusive coverage is substantial where peers miss attacks. Recalibrated ablations do not give additive causal attributions.
- Gains are concentrated but are not solely a TON_IoT result: ex-TON primary gains are 0.55–1.02 points; outcome-selected removal of the five largest gains leaves positive descriptive pooled intervals (M1).

**Readiness:** not ready for submission until C1 is completed. Scientific revision should first report the complete arm set (M13), bound the interpretation of sweep adjustment (M12), explain singleton conditioning (M14) and substantiate design/arm chronology (M16). No confirmed algorithmic defect or wrong primary printed result was found. Essential reporting corrections require no retraining. External comparison and independent targets would add evidence and may be required by the chosen journal; they are not demonstrated implementation repairs.

## 2. Priority Action Dashboard

| Priority | Finding | Required action | Change type | Effort | Expected benefit |
|---|---|---|---|---|---|
| P0 | C1 | Complete author-approved declarations and release/access facts | Manuscript only | Author-dependent | Submission responsibility |
| P1 | M13 | Report all rules, especially peers-only and reserve=.25 | Manuscript; numbers verified | 0.5–1 day | Complete comparative evidence |
| P1 | M12 | Explain clamping; remove exact-FPR/zero-credit-bound inference | Manuscript only | 1–2 h | Correct statistical interpretation |
| P1 | M14 | State singleton conditioning and host-level budget sensitivity | Manuscript; optional figure work | 0.5 day | Honest interval interpretation |
| P1 | M16 | Substantiate design/criteria/arm chronology | Manuscript only | Author-dependent | Transparent selection history |
| P1 | M5 | Move peer-fusion evidence into results with its FPR cost | Manuscript only | 1–2 h | Sharper contribution |
| P1 | M1, M20 | Add concentration and bank-budget-reallocation details | Manuscript only | 2–3 h | Defensible attribution |
| P1 | M2 | Choose narrow internal-study scope or one external reimplementation | Manuscript / optional experiment | 1 h / 1–2 weeks | Addresses closest-baseline objection |
| P2 | M17–M19; m15,m16,m17,m20,m21 | Add selected diagnostics and count/coverage clarifications | Manuscript; calculations verified | 0.5–1 day | Interpretable heterogeneity |
| P2 | m17, m13 | Identify exact datasets and figure generators | Manuscript; later release bookkeeping | 0.5 day | Reproducibility |
| P2 | O4, O6, O7, O9 | Add a small claim-specific literature selection | Manuscript only | 2–3 h | Novelty and current context |
| P2 | W7 | Clarify the first evidence-transform sentence | Manuscript only | 10 min | Method readability |
| P3 | Section 6 | Optional formatting, release notes, inset and wording | Later manuscript/figure/docs edits | As needed | Polish |


## 3. Critical Findings — P0

### C1 — Complete author-approved submission declarations

**ID:** C1 · **Priority:** P0 · **Category:** Reproducibility

**Location:** manuscript/main.tex:17,25,35–42; PDF pp. 1,25–26.

**Existing content or behavior:** Author identity, funding, conflicts, contributions, archive/access statements and final human approval remain expressly pending.

**Evidence:** The source and PDF contain the quoted placeholders. The AI statement explicitly names OpenAI Codex. A .claude directory alone cannot establish what another tool contributed; this audit cannot attest to the authors' complete preparation history.

**Issue:** Administrative and responsibility statements are incomplete. Incompleteness is confirmed; an undisclosed specific tool is not established.

**Why it matters:** The submission must identify responsible authors and contain accurate declarations. This is the only confirmed P0; no demonstrated algorithm defect is classified P0.

**Recommended fix:** Authors supply affiliations, contributions, funding/conflicts, data access/licences and an immutable code/results archive; confirm the applicable journal AI policy and describe actual relevant use and human review. Remove the draft banner only after approval. Do not invent a public release or require disclosure of a tool merely from directory names.

**Exact replacement:** Replace each pending statement with author-supplied facts. No factual authorship or approval declaration can responsibly be written by the auditor.

**Change type:** Manuscript only

**Verification:** Authors approve the final statements; all pending declarations are resolved; the stated archive is publicly accessible if public availability is claimed.

**Confidence:** Confirmed


## 4. Major Improvements — P1

### M13 — Report the complete executed comparison set

**ID:** M13 · **Priority:** P1 · **Category:** Scientific / Statistical

**Location:** §5.3, sections/experimental-methodology.tex:31–35; §6; results/tables/ablations.csv; experiment/arms.py:208–248.

**Existing content or behavior:** Methods define peers-only, local-only, reserve-0.25, recorded max-fusion and recorded Bonferroni rules. Results emphasize selected component and fusion contrasts.

**Evidence:** Independent JSON aggregation reproduces every arm. Peers-only TPR/FPR is 0.812411/0.030775 at α=.01 and 0.883945/0.066227 at α=.05. At α=.05 the primary comparator minus peers-only is −0.012597 [−0.022410,−0.004515] and −0.010327 [−0.019332,−0.003031]. PACE minus peers-only is +0.010821/+0.012892, with intervals crossing zero, accompanied by FPR changes −0.010774/−0.011890 whose benchmark intervals exclude zero. Full numerical inventory: §9.

**Issue:** Unreported executed arms leave a material completeness gap and conceal an informative simpler comparison. The scoped primary comparator remains legitimate for the complete local-branch replacement; its reported improvement is not numerically invalid.

**Why it matters:** A reviewer should not have to discover the peers-only tradeoff in released JSON. The favorable joint α=.05 means strengthen the paper without claiming resolved superiority to peers-only.

**Recommended fix:** Add an appendix table containing PACE, all 12 study arms and both recorded baselines, with TPR, FPR, PACE-minus-arm interval and precise arm definitions. In §6.1 report peers-only. Report reserve=.25 even where it detects more. For union scaling vs fixed shares, report the FPR cost; no Bonferroni sweep establishes equal-FPR superiority.

**Exact replacement:** Insert: “Giving the peers the entire nominal allocation yields detection 0.812411 at α=0.01 and 0.883945 at α=0.05. PACE improves on these means by 0.021–0.027 and 0.011–0.013, respectively, although all four paired detection intervals include zero. At α=0.05, PACE also has lower mean FPR by 0.010774 and 0.011890.”

**Change type:** Manuscript only (existing-results calculations verified here)

**Verification:** Every executed rule is reported once; peers-only is distinguished from the half-budget peer branch; no superiority claim exceeds its operating-point evidence.

**Confidence:** Confirmed

### M12 — Treat all clamped sweep estimates as approximate; zero-credit clipping is not a bound

**ID:** M12 · **Priority:** P1 · **Category:** Statistical

**Location:** §5.4 and §6.5, sections/results.tex:61–64; reporting/metrics.py:225–235.

**Existing content or behavior:** The manuscript correctly calls the four-point results “clamped sweep-adjusted comparisons” and reports uncovered endpoints. It does not describe the direction of endpoint bias.

**Evidence:** The primary gains 0.024659,0.025735,0.022176,0.020718 reproduce exactly. Below/above coverage counts are 47/28,49/20,12/24,9/19 out of 174. Below coverage, clamping credits detection measured at higher FPR. Clipping positive differences there to zero yields 0.008625,0.005115,0.021710,0.020442, but zero is not a lower bound on an unmeasured difference: if comparator TPR=.8 and the matched PACE TPR is .6, the difference is −.2. Allowing below-range PACE detection anywhere from zero to its observed endpoint produces conservative descriptive lower endpoints −0.210494,−0.220850,−0.043560,−0.027060 (remaining cells left at the shipped estimates). These are sensitivity calculations, not confidence guarantees or model evaluations.

**Issue:** Neither the four-point interpolation nor the zero-credit calculation demonstrates equal-FPR superiority. In particular, a positive zero-credit interval at α=.05 cannot establish robustness to all permissible below-range outcomes.

**Why it matters:** This changes the strength of the inference, not the nominal benchmark result. Endpoint uncertainty persists at both alpha levels. Even inside coverage, linear interpolation represents a mixture of stored decisions, not necessarily a deterministic threshold PACE actually tested.

**Recommended fix:** State the bias direction. Describe any clipped result as a “zero-credit sensitivity”, with its assumption explicit; do not call it pessimistic, a lower bound, exact matching or proof of robust matched-FPR superiority. Retain the nominal joint TPR/FPR comparisons. A bracketed matched-operating-point campaign is optional unless such superiority becomes a central claim (E1).

**Exact replacement:** Append to §6.5: “Below-sweep clamping uses PACE detection obtained at a higher realized FPR than the comparator, whereas above-sweep clamping uses detection obtained at a lower rate. Uncovered points leave the matched operating comparison unidentified. These estimates therefore support a sensitivity analysis of the nominal result, rather than establishing superiority at exactly equal FPR.”

**Change type:** Manuscript only; optional existing-results sensitivity

**Verification:** All uses of “matched”, “robust” and “bound” distinguish measured coverage, interpolation assumptions and uncovered cases; no zero-credit statistic is presented as a mathematical lower bound.

**Confidence:** Confirmed

### M14 — Explain singleton-stratum conditioning in the high-resource intervals

**ID:** M14 · **Priority:** P1 · **Category:** Statistical / Presentation

**Location:** §5.4; §6.6, results.tex:71,77; Figure 4 p.22; reporting/metrics.py:64–103; results/tables/budget-effects.csv.

**Existing content or behavior:** The 15-target cohort has corpus-stratified bootstrap bars and the N=100→20,000 α=.05 difference −0.028296 [−0.032827,−0.023766].

**Evidence:** The cohort comprises CTU-13 2, Gotham 2, IoT-23 2, N-BaIoT 3, TON_IoT 1, UNSW 5. A singleton contributes the same value to every bootstrap replicate. TON host 192.168.1.190 changes by −0.362268 at α=.05 and +0.283870 at α=.01. Removing it gives −0.004441 [−0.009296,0.000412] and +0.001791 [−0.003354,0.008440]. The printed interval is arithmetically correct for the stated conditional resampling procedure.

**Issue:** Conditional bars do not represent uncertainty in the singleton corpus contribution. Presenting the narrow pooled interval without this immediate explanation invites a broader inferential reading than the design supports.

**Why it matters:** The effect is an observed within-cohort result, largely one host. It is neither proof that more data generally hurts nor evidence that all bootstrap computations are wrong.

**Recommended fix:** Add the host-level and leave-one-host-out values. Either keep the bars with an explicit singleton caveat or show descriptive per-target changes; an unstratified target bootstrap can be a sensitivity, with its changed estimand/assumptions stated. Do not impose an arbitrary universal minimum stratum size or silently replace the registered procedure.

**Exact replacement:** Replace the high-resource interpretation with: “The α=0.05 change is dominated by TON_IoT host 192.168.1.190 (−0.362268); without it, the other 14 targets average −0.004441 [−0.009296,0.000412]. TON_IoT contributes a singleton to this cohort, so corpus-stratified resampling cannot vary its contribution. The conditional interval should not be read as population-level evidence that increasing local fitting data reduces detection.” Add to Table 4's caption: “Intervals condition on these corpora and seed-averaged targets; shared peers and new deployment populations are not resampled.”

**Change type:** Manuscript only; figure regeneration optional

**Verification:** The caption and results state the conditional estimand and singleton limitation next to the bars; leave-one-host-out numbers reproduce.

**Confidence:** Confirmed

### M16 — Document design selection and comparison-arm timing precisely

**ID:** M16 · **Priority:** P1 · **Category:** Methodological / Reproducibility

**Location:** §5.5, experimental-methodology.tex:50; §8; docs/PACE.md §§8,11.

**Existing content or behavior:** The manuscript already discloses reuse of development targets and says held-out seeds assess a fixed design. Project documentation reports seeds 6–8 were unused for final selection and that an earlier failure on IoT-23 35-1 motivated the anchor resolution rule.

**Evidence:** The executed files use seeds 6,7,8. docs/PACE.md describes the earlier 35-1 failure and selection of pair/response components. The public Git record introduces the arm code together on 10 October; it does not independently establish the exact sequence of private development runs or the alleged post-confirmation arm additions. Those timing claims require author-confirmed dated records.

**Issue:** The key remaining gap is selection detail and verifiable chronology, not absence of any disclosure. Untouched randomness on reused targets does not make the target outcomes independent of model design.

**Why it matters:** Reviewers can reasonably request which outcomes informed the architecture, acceptance margins and reported contrasts. A clear account is stronger than an unverifiable claim of preregistration.

**Recommended fix:** Add a short author-confirmed development paragraph: data/cohort reused; design decisions informed by development outcomes; final freeze and acceptance criteria timing; status of seeds 6–8; any arms added after outcomes. Archive supporting timestamps where shareable. Do not assert exact dates, seed ranges 0–5 or post-hoc status solely from a previous review.

**Exact replacement:** Safe text now: “The same target cohort informed development. An earlier failure on IoT-23 capture 35-1 motivated the anchor resolution rule. The project specification identifies seeds 6–8 as unused for final design selection; their role is to assess learning-randomness stability on reused targets. The authors will document the design freeze, acceptance-criterion freeze and timing of comparison-arm additions in the release.” Replace the final sentence with verified facts before submission.

**Change type:** Manuscript only

**Verification:** The authors substantiate freeze/arm timing with dated evidence; confirmation and exploratory contrasts are labeled accurately.

**Confidence:** Confirmed for documented reuse and design motivation; Unresolved for exact chronology

### M5 — Give the peer-fusion contrast appropriate prominence and retain its alert-rate cost

**ID:** M5 · **Priority:** P1 · **Category:** Scientific / Presentation

**Location:** Abstract; Introduction contribution 2; §7.2, discussion.tex:13; experiment/arms.py:208–229.

**Existing content or behavior:** The headline quantifies a local-branch replacement; the method-specific peer-wise-versus-fused result appears in the discussion.

**Evidence:** With PACE's local bank fixed, nominal gains over mean fusion are .073437/.055978 at α=.01 and .052459/.044723 at α=.05; over max fusion .045555/.036390 and .030484/.031644. All eight descriptive paired intervals exclude zero. PACE FPR exceeds the peer-max arm by .008872,.008514,.009022,.011288. Clamped adjusted max-fusion gains at α=.05 are .017934 [.006094,.031205] and .019003 [.007637,.031730], subject to M12. Per-peer alert evidence is not retained, so the specific drowning mechanism is not empirically isolated.

**Issue:** The title's architectural contribution deserves a results subsection. However, nominal gains are a detection/false-alert tradeoff, and the adjusted numbers do not prove exact matched-FPR superiority.

**Why it matters:** This is stronger positioning than claiming novelty for known estimators, while avoiding an unsupported causal attribution to drowning.

**Recommended fix:** Move the fusion numbers into §6, retain their FPRs, and cross-reference from §7.2. Keep the four primary local-branch results. Verify comparison-arm timing under M16 before calling these confirmatory. Do not replace the abstract with a catalogue of all exploratory findings.

**Exact replacement:** Suggested abstract addition: “With the local bank fixed, separate peer thresholds improve detection by 3.0–7.3 percentage points over mean or maximum peer fusion at the tested nominal settings, with higher realized false-alert rates.” Replace the last §7.2 sentence by: “The peer-wise architecture trades higher detection for higher mean FPR at the tested nominal settings; the clamped sweep estimates provide additional, approximate operating-point sensitivity evidence.”

**Change type:** Manuscript only

**Verification:** The result appears in §6 with its fixed components, FPR cost and exploratory/confirmation status; no causal or exact-FPR claim exceeds the evidence.

**Confidence:** Confirmed

### M1 — Quantify concentration while retaining the valid cross-corpus result

**ID:** M1 · **Priority:** P1 · **Category:** Scientific / Statistical

**Location:** Abstract; §6.3, results.tex:27–31; §7.3; results/tables/targets.csv and acceptance.csv.

**Existing content or behavior:** Gains are described as strongest in TON_IoT and smaller elsewhere; the four ex-TON intervals are printed correctly, followed by “two lower bounds lie very close to zero”.

**Evidence:** Ex-TON means are .009306,.010228,.005536,.008299; lower bounds .001808,.000995,.000090,.000143. Three are below .001; “very close” alone is subjective. The top three target effects supply about 60–89% of the pooled sum. Removing the five largest-gain targets leaves .006295,.006253,.001605,.001055, each with a positive conditional interval. This deletion is outcome-selected and descriptive. IoT-23's gain is dominated by capture 3-1.

**Issue:** Give concentration a quantitative interpretation, without treating the remaining positive means as a broad population guarantee or declaring accurate aggregate statements false.

**Why it matters:** A reviewer skeptical of TON_IoT dependence gets an honest answer: substantial gains on a minority, small positive pooled residuals, and genuine negative subgroup outcomes retained.

**Recommended fix:** Replace the subjective lower-bound count with exact wording; add top-target contribution or leave-top-five-out as exploratory sensitivity. Keep the 750/2000-row negative or uncertain ex-TON findings. Known benchmark-label smells motivate checking the evaluated subsets, not declaring all TON results invalid or all artefacts absent.

**Exact replacement:** Replace: “The remaining gains are positive but small (0.55–1.02 percentage points); three of the four interval lower bounds are below 0.001. Three targets contribute approximately 60–89% of the summed gains. As an outcome-selected sensitivity analysis, removing the five largest-gain targets leaves pooled gains of 0.0011–0.0063.”

**Change type:** Manuscript only

**Verification:** All four intervals and adverse larger-budget results remain; the sensitivity is labeled outcome-selected, with target counts and weighting explicit.

**Confidence:** Confirmed

### M20 — Describe the anchor ablation as a bank-budget reallocation with target-specific benefits

**ID:** M20 · **Priority:** P1 · **Category:** Scientific / Statistical

**Location:** §6.2, results.tex:22; §7.1, discussion.tex:8; experiment/arms.py:170–185.

**Existing content or behavior:** The current discussion accurately describes a tradeoff: removing the anchor raises detection and FPR at α=.05. At α=.01 its nominal benefit is unresolved.

**Evidence:** BANK_WITHOUT_ANCHOR puts the bank alone at the full local allocation B_L, rather than holding its budget B_B fixed. PACE-minus-arm nominal differences are +.010092 [−.017927,.052456],+.014899 [−.009737,.054637],−.002832 [−.005608,−.000786],−.001535 [−.003242,−.000329]. Capture 35-1 contributes +1/58 at α=.01, while TON .190 loses .462/.310. The clamped adjusted differences −.013291/−.007955 at α=.01 remain approximate (M12).

**Issue:** The target-specific nominal gain should be shown, but neither this reallocation ablation nor approximate adjustment isolates a universal “anchor effect”. A “worst-case guard” is an interpretation, not a proved worst-case protection.

**Why it matters:** The component is explicitly named and was selected in response to 35-1. Transparent attribution avoids overvaluing it or recommending its deletion from one exploratory metric.

**Recommended fix:** Name the budget reallocation, report 35-1 and the compensating host loss, retain the α=.05 tradeoff. Relate the motivating development example to M16. Do not claim exact matched-FPR inferiority or non-inferiority protection.

**Exact replacement:** Insert after the first anchor sentence: “This ablation reallocates the full local budget to the bank, so it changes both anchor presence and bank allocation. At α=0.01, capture 35-1 contributes 0.01724 to the pooled difference, offset by losses including TON_IoT host 192.168.1.190; the net intervals include zero. The example was also a development motivation for the resolution rule.”

**Change type:** Manuscript only

**Verification:** The ablation definition includes budget changes, nominal results remain intact, and approximate adjustments are not labeled exact matching.

**Confidence:** Confirmed

### M2 — Prioritize one protocol-compatible external baseline if broader comparative claims are desired

**ID:** M2 · **Priority:** P1 · **Category:** Scientific

**Location:** Table 1; §8 “Comparison and attribution”; configs/default.yaml:study_arms.

**Existing content or behavior:** All executed alternatives are internal. The paper explicitly says external systems were not rerun and claims no measured superiority over them.

**Evidence:** The campaign contains 12 named internal arms plus two recorded baselines. MCDE has verified algorithmic overlap and uses IoT-23/N-BaIoT, but its principal task is family hold-out and its supervised source model differs from PACE's peer bank. The local-only arm is a useful internal benign-only baseline, not a substitute for all contemporary methods.

**Issue:** A demanding journal reviewer can still ask whether the architectural gain advances over the closest published approach. The disclaimer resolves overclaiming, not the empirical comparison itself.

**Why it matters:** External comparison is a likely acceptance risk whose severity depends on venue and the breadth of the final claim. It does not invalidate a clearly scoped internal architectural study.

**Recommended fix:** If a new experiment is feasible, implement a documented MCDE adaptation under exactly the PACE target roles and source access; report all results and deviations, both detection and FPR, with independently reserved evaluation randomness/data where possible. Do not describe a modified reimplementation as the original published result. Otherwise retain the limitation and precise architecture-only claims.

**Exact replacement:** Optional §8 addition: “A protocol-compatible reimplementation of MCDE on shared corpora would test the benefit against a close published hybrid; the present comparisons isolate internal design choices and do not answer that external-superiority question.”

**Change type:** New experiment optional; Manuscript only for scoped claims

**Verification:** No cross-paper numerical leaderboard is implied; an external run, if later authorized, has an explicit compatibility/adaptation table.

**Confidence:** Confirmed


## 5. Minor Improvements — P2

### M17 — Use existing diagnostics to localize false alerts without claiming causal identification

**ID:** M17 · **Priority:** P2 · **Category:** Scientific

**Location:** §6.4, results.tex:50; cache manifests and benign arrays; reporting/metrics.py aggregation.

**Existing content or behavior:** The paper correctly says outcomes do not isolate temporal change, sampling or extrapolation as causes of FPR heterogeneity.

**Evidence:** Target FPR has Spearman correlation approximately −.58 to −.52 with benign test size (a near-equal proxy for calibration size), confounded with corpus. In IoT-23 7-1, 65.8% of test rows lie outside the calibration range of orig_ip_bytes; PACE and comparator both reach about .67. In UNSW G2, a repeated benign vector contributes about 22% of test rows but is absent from calibration (M19). CTU-13 scenario 11's .1912 FPR is also present in the separately thresholded peer branch and much smaller under fused peers; this localizes a branch, not a responsible peer or causal covariate.

**Issue:** The existing data support a useful diagnostic paragraph. Correlation and range changes do not establish which scores caused each alert, exclude extrapolation, or prove that a larger calibration set would repair deployment.

**Why it matters:** Operational interpretation improves if extreme rates are investigated rather than merely listed. Shared failures remain operationally serious even when all methods suffer them.

**Recommended fix:** Retain the cautious causal sentence and add role-shift, point-mass and size associations. Use the exact calibration-role counts if plotting size. Distinguish delayed/noncausal Gotham features, separately disclosed in §5.1, from target-label leakage. Per-peer re-scoring is optional future work, not performed here.

**Exact replacement:** Append: “Read-only diagnostics associate higher rates with smaller calibration roles and substantial benign role changes: on capture 7-1, approximately 66% of test observations lie outside the calibration range of originator IP bytes. These observations localize possible distribution effects; they do not identify a causal detector or show that increasing calibration data alone would restore nominal control.”

**Change type:** Manuscript only; optional existing-results figure

**Verification:** Numbers use chronological roles; plots identify confounding and distinguish associations from score-level causal evidence.

**Confidence:** Confirmed

### M19 — Explain the mean-FPR reduction without weakening the correct headline

**ID:** M19 · **Priority:** P2 · **Category:** Statistical

**Location:** Abstract; §6.1; Table 4; highlights.txt:3.

**Existing content or behavior:** PACE has lower mean FPR at α=.05 at both budgets. The text explicitly labels the result aggregate and says individual targets need not improve on both measures.

**Evidence:** FPR differences are −.001984 [−.003909,−.000189] and −.010070 [−.021626,−.001486]. At N=400, targets d073d5018308 and 74c63b29d71d contribute about .0080 of the .0101 reduction. Excluding them still gives −.002107 [−.003869,−.000655]. Lower/higher/equal target counts are 29/23/6 and 32/23/3. Each target has an approximately 22% test point mass absent from calibration; it occurs once, not zero times, in each reference role. Alert masks/score traces are absent, so this is a strongly supported association rather than proved causation.

**Issue:** The magnitude is concentrated, but the aggregate claim is accurate and the reduction is not eliminated by excluding the two targets. Calling the headline false or mandating “slightly” would unnecessarily weaken valid evidence.

**Why it matters:** A compact target-level explanation makes the favorable joint TPR/FPR result more persuasive and gives readers its limits.

**Recommended fix:** Keep the exact headline and highlights. Add FPR-difference intervals to Table 4 and one sentence on the two-target contribution. Do not claim typical-device dominance from the mean or causal attribution from rates alone.

**Exact replacement:** Insert in §6.1: “At N=400, two UNSW G2 targets contribute about 0.0080 of the 0.0101 mean-FPR reduction; excluding them leaves a reduction of 0.002107 [0.000655,0.003869]. Across the 58 targets, 32 have lower FPR, 23 higher and three equal.”

**Change type:** Manuscript only

**Verification:** The exact aggregate improvement remains; concentration, exclusion result and target counts agree with stored rates.

**Confidence:** Confirmed

### M18 — Show win/tie/loss and detection headroom descriptively

**ID:** M18 · **Priority:** P2 · **Category:** Statistical / Presentation

**Location:** §6.3 and Figure 2; results/tables/targets.csv.

**Existing content or behavior:** The text appropriately cautions that positive mean gains do not mean most devices improve.

**Evidence:** Using an exploratory tolerance ±.001 after seed averaging, wins/ties/losses are 16/42/0,21/36/1,14/42/2,15/41/2 in primary order. At α=.01,N=400, 32 comparator-saturated targets (miss rate≤.01) gain about .0001, while the other 26 gain about .0630. A sign test would condition on the chosen tolerance and treat shared-peer targets as independent; it does not resolve the documented dependence.

**Issue:** The useful structure is sparse positive effects with few material losses. Calling this “no harm” is too broad, and adding small p-values cannot turn benchmark targets into independent deployments.

**Why it matters:** Counts show complementarity and ceiling effects more clearly than a median of zero.

**Recommended fix:** Add the win/tie/loss counts and label the tolerance exploratory; optionally show the headroom split as outcome-conditioned description. A sign-test p-value is unnecessary; if retained, explicitly state its independence assumption and lack of multiplicity correction.

**Exact replacement:** Append: “At α=0.01 and N=400, 21 targets gain more than 0.001, 36 differ by at most 0.001, and one loses more than 0.001; at α=0.05 the corresponding counts are 15,41,2. The tolerance is descriptive and the targets share peer models.”

**Change type:** Manuscript only

**Verification:** Counts reproduce from seed-averaged target effects; adverse target outcomes remain visible.

**Confidence:** Confirmed

### m20 — Connect the allocation-respecting single-union result to the low-alpha budget exception

**ID:** m20 · **Priority:** P2 · **Category:** Methodological

**Location:** §4.5, methodology.tex:81; §6.2; experiment/arms.py:112–123.

**Existing content or behavior:** The manuscript discloses component allocations summing to .0125 at α=.01 and separately reports identical pooled detection for the single-union arm.

**Evidence:** BANK_SINGLE_UNION calibrates all local experts jointly within B_L=.005, so total allocation is .01. Its TPR equals PACE's at stored precision in both primary budgets; FPR is .033998/.032371 versus .033992/.032369. Aggregate per-type TPR and FPR summaries agree in 344/348 target–seed cells. JSON stores aggregate rates, not per-row decisions, so identical alert masks cannot be inferred. Four actual single-block targets allow a bank union alert at B_B=.0025, so “the bank can permit no extra calibration exceedance” is not universal.

**Issue:** This is useful evidence that a strictly allocation-respecting alternative retains the observed pooled detection, not a proof that the extra allocation is causally irrelevant or that the two rules alert on identical rows.

**Why it matters:** It answers the 25%-larger-allocation objection without changing the rule or concealing the exception.

**Recommended fix:** Place the single-union result next to the allocation exception and use rates rather than mask-identity language.

**Exact replacement:** Append after “Their sum is 0.0125, rather than 0.01.”: “The single-local-union ablation instead respects total allocation α and has the same pooled detection at α=0.01 to the precision of the stored results, with very similar mean FPR (Section 6.2). This is an empirical sensitivity comparison, not a finite-sample FPR guarantee.”

**Change type:** Manuscript only

**Verification:** The single-union allocation follows arms.py; exact equality is qualified by the retained summaries and does not assert alert-mask identity.

**Confidence:** Confirmed

### m15 — Explain individual-quantile resolution separately from the joint union count

**ID:** m15 · **Priority:** P2 · **Category:** Methodological

**Location:** §4.2–4.5; core/decision.py:56–81,276–311; five single-block target roles.

**Existing content or behavior:** The current manuscript already distinguishes the heuristic 1/b from the actual higher-quantile resolution 1/(n_j−1).

**Evidence:** For a single unscaled threshold at w, strict exceedances are at most floor(w(n_j−1)). For the empirically checked union the allowed count is floor(B n_j). With B=.005 a 250–399-row block permits one union exceedance and a 400–499-row block permits two. With B=.0025, blocks below 400 permit none and blocks≥400 permit one. Actual single blocks are 385,495,436,464,418 rows; four thus allow two peer/anchor and one bank exceedance. For an individual threshold at w=.0025, first nonzero exceedance is at n=401, a different boundary.

**Issue:** A blanket “one exceedance per block” description is incorrect for the actual longer single blocks. The paper's equation itself is correct.

**Why it matters:** Integer resolution explains flat sweeps and why nominal allocation differences need not yield smooth operating changes.

**Recommended fix:** If adding explanatory count language, state both formulas and their different boundaries; keep the existing heuristic disclaimer. No algorithm change is warranted.

**Exact replacement:** Optional §4.5 sentence: “The joint criterion permits floor(B n_j) alerts per block, whereas an individual unscaled higher quantile admits at most floor(w(n_j−1)) strict exceedances; these are different resolution limits.”

**Change type:** Manuscript only

**Verification:** Check n=399,400,401 and the actual five single-block sizes against both formulas.

**Confidence:** Confirmed

### m16 — Quantify seen-label transfer without equating label names with behavior

**ID:** m16 · **Priority:** P2 · **Category:** Scientific

**Location:** §5.2, experimental-methodology.tex:19; cache manifests; peer_model_seeds in campaign JSON.

**Existing content or behavior:** Peers may have seen target attack families; the paper disclaims uniform unseen-family evaluation.

**Evidence:** Reconstructing eligible development types for each actual foreign peer gives 11/258 evaluated target–attack-type cells with a label absent from every peer; 247 have a peer label match. Minima 400/600/30 and actual peer-only clients were applied. Names such as generic C&C do not establish identical behaviors, and source-generated attacks can be correlated.

**Issue:** “May also appear” can be more informative quantitatively. This is label overlap, not proof of attack instance leakage.

**Why it matters:** Separates benign-only target access from zero-day-family claims and clarifies comparison to MCDE.

**Recommended fix:** Add the 11/258 count, with the label-based definition; keep the existing restriction on generalization.

**Exact replacement:** Replace the clause with: “including attack labels that also occur at the target; only 11 of 258 evaluated target–attack-type pairs have labels absent from every eligible peer's development data. Label overlap does not imply identical attack behavior.”

**Change type:** Manuscript only

**Verification:** Coverage is reconstructed from actual peers and eligible split types; no family-holdout claim is introduced.

**Confidence:** Confirmed

### m21 — Localize the scenario-12 regression without declaring it nonsystematic

**ID:** m21 · **Priority:** P2 · **Category:** Scientific

**Location:** §6.3, results.tex:31; CTU13-scenario-12 primary records.

**Existing content or behavior:** The paper reports the −.032595 target loss and correctly says the full branch comparison does not identify one detector as its cause.

**Evidence:** At α=.01,N=400, seed 7 contributes about −.095 (PACE .218 vs comparator .313), while seeds 6/8 are about 0/−.003. Its seed-7 peer union detects .189, versus .997/.986 in the other seeds. At N=100 that seed favors PACE (.575 vs .232). NSIS.AY is among the 11 absent-peer-label cells.

**Issue:** These records localize the observed loss to a seed/budget combination. Three seeds cannot establish that the failure is “not systematic”, and an autoencoder-specific cause is not identified.

**Why it matters:** This is more useful than either suppressing the regression or attributing it to an unmeasured component.

**Recommended fix:** Retain the current causal caution; optionally add the seed-level evidence and expose all per-seed outcomes in supplementary results.

**Exact replacement:** Append: “Most of the seed-averaged loss occurs in seed 7, where peer detection is 0.189 and the comparator's local route recovers more at N=400. The same seed favors PACE at N=100. These observations localize the reported regression but do not identify its detector-level cause.”

**Change type:** Manuscript only

**Verification:** Per-seed numbers reproduce; the regression remains labeled a regression and no untested detector cause is asserted.

**Confidence:** Confirmed

### m17 — Identify the exact dataset releases and access conditions

**ID:** m17 · **Priority:** P2 · **Category:** Reproducibility / Literature

**Location:** §5.1; main.tex:35–36; references.bib dataset entries; cache manifest source_file fields.

**Existing content or behavior:** The dataset names and citations are present, but submission availability is pending and archives/subsets are not fully identified.

**Evidence:** Manifest paths identify IoT-23-v2/iot_23_datasets_small.tar.gz, TON-IoT/Processed_datasets/Processed_Network_dataset, Gotham processed device CSVs, per-device UNSW flowstats, N-BaIoT and CTU-13. DataCite verifies IoT-23 DOI 10.5281/zenodo.4743746 (v1.0.0) and Gotham DOI 10.5281/zenodo.14502760 (2025); these records carry CC BY 4.0. N-BaIoT dataset DOI is 10.24432/C5RC8J, distinct from the Meidan paper. The 2026 Gotham descriptor DOI 10.3390/data11080207 cites the same deposit.

**Issue:** A concept/data DOI is not enough to identify a particular archive, local extraction or downstream preprocessing. The raw multi-gigabyte archive checksum was not independently recomputed in this audit; do not present inherited checksums as newly checked.

**Why it matters:** A reproducer needs the same inputs and legal access, not a generic link to another archive with a similar name.

**Recommended fix:** Add exact source URLs, archive/subset/version, manifest or checksum already verified by the release process, and per-corpus terms. Cite the N-BaIoT data DOI and the Gotham descriptor alongside the existing paper/deposit. For IoT-23 distinguish the labelled-flow small archive from full PCAP archives. Do not claim that the existing dataset DOI is a wrong citation simply because a separate small archive was used.

**Exact replacement:** Data-availability template: “We used [exact archive/subset], retrieved from [authoritative URL/version], with integrity identifiers in [release manifest]. The corpus-specific preprocessing is described in Section 5.1. Access and redistribution terms are [per-corpus terms].” Fill these brackets with verified release facts.

**Change type:** Manuscript only

**Verification:** A reproducer can locate each exact input and verify the stated identity; licences are checked for the actual files, not inferred across all corpora.

**Confidence:** Confirmed

### m13 — Include the generators for the manuscript-specific figures

**ID:** m13 · **Priority:** P2 · **Category:** Reproducibility

**Location:** manuscript/figures/{architecture,target-gains,false-alert-distribution,budget-scaling}.pdf; reporting/figures.py:350–369.

**Existing content or behavior:** The repository has campaign-report generators, but the manuscript uses different figure PDFs and an architecture asset.

**Evidence:** Manuscript and results/figures versions have different hashes and layouts. The displayed label “Target rank (sorted within panel)” is not found in repository figure source. No repository generator for architecture.pdf was found. This establishes an absent reproducibility route in the inspected tree, not misconduct or externally copied figures.

**Issue:** The publication's exact visual assets cannot be regenerated from the documented code path.

**Why it matters:** Authors and reviewers need to reproduce the submitted plots and apply M14 without hand editing PDFs.

**Recommended fix:** At release, add the manuscript-figure source and an input manifest outside the frozen campaign implementation scope; document font/library versions and the command. Do not alter the PACE implementation to regenerate presentation-only figures.

**Exact replacement:** No prose replacement required; availability should name the figure-generation command and inputs.

**Change type:** Code modification (figure/release bookkeeping only, later); Manuscript only for documentation

**Verification:** Running the documented figure-only command in a temporary output directory reproduces submitted content without training or overwriting frozen evidence.

**Confidence:** Confirmed

### O4 — Add focused evaluation-practice sources where they do specific work

**ID:** O4 · **Priority:** P2 · **Category:** Literature

**Location:** §7.4 operational interpretation; §8 population/dependence; §§10–11 of this audit.

**Existing content or behavior:** Sommer/Paxson and Arp already provide sound evaluation context.

**Evidence:** Apruzzese 2026 (DOI 10.1145/3779208.3807488) discusses operational reports/triage; Apruzzese et al. 2023 (10.1109/EuroSP57164.2023.00042) supplies pragmatic NIDS evaluation context. Flood et al. 2024 (10.1109/EuroSP60621.2024.00042) is corroborated through the author's Edinburgh thesis chapters 3–4, not a downloaded conference PDF. TESSERACT has an official Eq.(4) erratum; it does not change its temporal/spatial-bias message.

**Issue:** Selected sources can sharpen the link between benchmark quantities and deployment. Adding every possible evaluation citation would be redundant.

**Why it matters:** The paper benefits from current security-methodology context without a longer catalogue of limitations.

**Recommended fix:** Prefer Apruzzese 2026 next to incident/row-alert distinctions and Flood next to dataset-quality limitations. Add TESSERACT only if explicitly discussing temporal/spatial bias, using its corrected archival version. No source proves that PACE's own evaluated rows are mislabeled.

**Exact replacement:** Suggested §8 clause: “Benchmark construction can introduce dataset-specific design artefacts [Flood et al., 2024]; their impact must be checked on the retained subsets rather than inferred from corpus names.”

**Change type:** Manuscript only

**Verification:** Each added citation supports its attached claim; the TESSERACT correction is acknowledged in reference notes if that source is added.

**Confidence:** Confirmed

### O6 — Include close transfer and collaborative-learning neighbors selectively

**ID:** O6 · **Priority:** P2 · **Category:** Literature

**Location:** §2.4, related-work.tex:25–31.

**Existing content or behavior:** DÏoT, FedMSE, personalized hypernetworks and OSDN are discussed accurately.

**Evidence:** Rey et al., Computer Networks 204:108693, DOI 10.1016/j.comnet.2021.108693, evaluates federated benign-only autoencoders and supervised classifiers on N-BaIoT, including unseen-device behavior. Layeghy et al., KBS 273:110626, DOI 10.1016/j.knosys.2023.110626, trains adversarial domain-invariant representations using labeled sources and unlabeled targets, then uses a one-class SVM. Their local PDFs and relevant methods were inspected. Rey's stored preprint includes Martin Jaggi in its author list; the final DOI record lists four authors.

**Issue:** These are closer information-regime comparisons than a generic deep-IDS catalogue, but neither implements PACE's fixed per-peer target threshold architecture.

**Why it matters:** Clarifies what is established and prevents benign-only/collaborative novelty from being overstated.

**Recommended fix:** Add one precise sentence per method, using final publication metadata while identifying preprint versions in the reference collection.

**Exact replacement:** Insert: “Rey et al. study federated benign-only autoencoders and supervised classifiers on N-BaIoT, whereas PACE retains distinct frozen supervised peers. DI-NIDS learns a domain-invariant representation from labeled sources and unlabeled target traffic before fitting a one-class SVM; PACE instead calibrates compatible fixed peer models using trusted target benign data.”

**Change type:** Manuscript only

**Verification:** Information access, target-data use and final decision differences are faithful to the full methods; no cross-paper performance ranking is implied.

**Confidence:** Confirmed

### O7 — Acknowledge alert-budget ensemble thresholding precedents

**ID:** O7 · **Priority:** P2 · **Category:** Literature / Scientific

**Location:** §2.2 and §2.5; Introduction paragraph 4.

**Existing content or behavior:** The paper correctly distinguishes empirical thresholds from conformal validity, but lacks an operational ensemble-thresholding precedent.

**Evidence:** Bridges, Jamieson and Reed 2017 (DOI 10.1109/BigData.2017.8258031) regulate alert rates across heterogeneous probabilistic detectors. CADES, Zhang et al., ICML 2025, PMLR 267:77177–77193, combines complementary conformal scores with Bonferroni correction. Relevant methods in both PDFs were read. CADES' official proceedings record supplies an archival identifier but no publication DOI was verified.

**Issue:** Per-detector thresholds under a total alert/FPR budget are established. PACE's empirical target-block joint scaling should be positioned as a particular transfer architecture, not invention of budgeted detector unions.

**Why it matters:** Direct prior-art acknowledgment makes the legitimate integration contribution more defensible.

**Recommended fix:** Add Bridges as the DOI-verified ensemble-threshold source; discuss CADES as a verified proceedings neighbor if the author accepts a non-DOI archival citation. Do not invent a DOI or claim PACE meets CADES' validity assumptions.

**Exact replacement:** Insert after the p-value-combination discussion: “Per-detector threshold selection to regulate the alert rate of heterogeneous detector ensembles has also been studied [Bridges et al., 2017]. PACE applies empirical joint threshold scaling to frozen foreign supervised experts on target benign blocks; it does not establish conformal p-value validity.”

**Change type:** Manuscript only

**Verification:** The novelty text reserves established union/threshold ideas for their sources and states the distinct transfer and calibration procedure.

**Confidence:** Confirmed

### W7 — Make the first evidence-transform description precise

**ID:** W7 · **Priority:** P2 · **Category:** Writing

**Location:** §1, introduction.tex:7.

**Existing content or behavior:** “PACE, Peer-wise Calibrated Experts, maps each supervised peer's margin relative to scores on a target benign reference role.”

**Evidence:** Equation (2) specifies tie-inclusive upper-tail ranking with a separate beyond-maximum extrapolation.

**Issue:** “Maps … relative to scores” is hard to parse and leaves the kind of transformation unspecified. A replacement should include extrapolation rather than imply every output is only a rank.

**Why it matters:** The first short method description should be intelligible before the equations.

**Recommended fix:** Replace only this sentence; leave technically precise formal prose alone.

**Exact replacement:** “PACE (Peer-wise Calibrated Experts) converts each peer's attack-minus-benign margin into target-relative upper-tail evidence, using empirical ranks on a benign reference role and extrapolation beyond its maximum.”

**Change type:** Manuscript only

**Verification:** The replacement matches Equation (2), including its beyond-reference branch; scientific meaning is preserved.

**Confidence:** Confirmed

### O9 — Use recent calibration-diagnosis work without importing its dataset-specific conclusions

**ID:** O9 · **Priority:** P2 · **Category:** Literature

**Location:** §2.6 after Ha/Kim; optional §8 calibration paragraph.

**Existing content or behavior:** The manuscript already emphasizes the distinction between ranking and threshold transfer; it has no reference to the newly located controlled calibration-granularity study.

**Evidence:** Nouman, Hassanu and Ujjan, Future Internet 18(9):447 (2026), DOI 10.3390/fi18090447, published 24 August, identifies a window-average versus per-flow scoring mismatch through controlled re-scoring. PDF §§3.7,4.7,5.2 were read; Proposed_14_Nouman_2026.pdf is the matching 45-page version of record. PACE instead uses row-level evidence at calibration and inference and takes maxima of block-specific quantiles; no analogous granularity mismatch was found.

**Issue:** The paper is relevant methodological context, not evidence that PACE's FPR is caused by the same fault. Another fresh candidate, MEFA (10.1002/spy2.70069), has verified metadata/abstract but unavailable full text; it must not be described as fully checked prior art.

**Why it matters:** A current concrete calibration example can strengthen the diagnostic argument while preserving PACE's actual contribution and uncertainty.

**Recommended fix:** Add Nouman only if a concise concrete example improves §2.6; do not add its numerical performance as a baseline. Keep MEFA in a full-text follow-up list, with no detailed algorithm claim beyond its abstract until obtained.

**Exact replacement:** Suggested sentence: “A controlled 2026 study attributes a large fixed-threshold difference to a mismatch between calibration and evaluation score granularity [Nouman et al.]; this illustrates why a calibration procedure must be assessed separately from its anomaly architecture.”

**Change type:** Manuscript only

**Verification:** The DOI and local PDF match; the text does not attribute the external study's failure or data-quality claims to PACE.

**Confidence:** Confirmed


## 6. Optional Enhancements — P3

### M10 — Describe repeated cohort records accurately

**ID:** M10 · **Priority:** P3 · **Category:** Reproducibility

**Location:** outputs/campaigns/*/*/results.json; availability/release manifest.

**Existing content or behavior:** Overlapping resource cohorts retain repeated executions of common target/seed/budget cells.

**Evidence:** 110 files contain 1,824 records and 1,416 unique (alpha,budget,stratum,target,seed) keys. All 408 repeats agree in every serialized target field. All content digests validate; the current implementation digest matches 7dd88ce13b0d32a3875db47869e5ac13f5cecfa0dc82c6d14322dc67419a8a3b.

**Issue:** Repeats are not independent observations or conflicting results.

**Why it matters:** A release manifest and explicit deduplication key prevent double counting.

**Recommended fix:** Add one availability sentence and a manifest of artifact paths; preserve all evidence.

**Exact replacement:** “Resource-cohort campaigns contain 408 repeated target–seed records at shared budgets; repeated records are identical and are counted once per analysis.”

**Change type:** Manuscript only

**Verification:** The manifest has 110 paths; aggregation uses the specified unique key.

**Confidence:** Confirmed

### m14 — Name the power-planning approximation

**ID:** m14 · **Priority:** P3 · **Category:** Statistical

**Location:** §5.5, experimental-methodology.tex:48; metrics.py:paired_minimum_detectable_effect.

**Existing content or behavior:** The planning calculation is already distinguished from observed power.

**Evidence:** NormalDist is used; .05×(1.959964+.841621)/sqrt(58)=.018393, and doubling assumed SD doubles MDE.

**Issue:** The normal approximation is implicit.

**Why it matters:** One phrase makes the calculation reproducible.

**Recommended fix:** Insert “under a normal approximation” after “planning calculation”.

**Exact replacement:** “The outcome-independent power table is a planning calculation under a normal approximation …”

**Change type:** Manuscript only

**Verification:** The normal quantiles reproduce the MDE.

**Confidence:** Confirmed

### m19 — Align release-facing documentation with the manuscript

**ID:** m19 · **Priority:** P3 · **Category:** Reproducibility

**Location:** README.md:5,9; docs/PACE.md §§1–2,5,9,13.

**Existing content or behavior:** README says no raw peer data are used and calls all 58 targets devices. docs/PACE.md additionally describes a strict partitioned budget, no neural training, and anchor non-inferiority.

**Evidence:** The standardizer pools benign training rows; targets include scenarios/hosts; allocations sum to .0125 at α=.01; the harness trains the comparator autoencoder; the anchor does not ensure comparator non-inferiority. The manuscript already states these correctly.

**Issue:** Stale supporting documentation could contradict an otherwise careful paper.

**Why it matters:** Release instructions should match the paper’s information assumptions.

**Recommended fix:** During a later authorized release edit, align these exact statements with §§3–5 of the manuscript. No documentation was edited here.

**Exact replacement:** README: “The target detector fits and calibrates without target attack labels; the centralized benchmark uses pooled benign preprocessing.” Use “58 benchmark targets”. In docs, replace guaranteed protection and strict-budget language with the implemented heuristic and empirical results.

**Change type:** Code modification (documentation only, later)

**Verification:** README/docs and manuscript agree on preprocessing, targets, allocation, neural fitting and lack of non-inferiority guarantee.

**Confidence:** Confirmed

### m22 — Apply bibliography style consistently without inventing substantive errors

**ID:** m22 · **Priority:** P3 · **Category:** Presentation

**Location:** references.bib; PDF pp.26–28.

**Existing content or behavior:** Some titles are brace-protected, others sentence case; MCDE prints DOI plus an identical URL; Pang uses article-based page numbering.

**Evidence:** The temporary build has no LaTeX layout or undefined-reference warnings; BibTeX emits one empty-pages warning for Kitsune. NDSS articles lack conventional page ranges. ACM article-style pages such as 38:1–38:38 are legitimate.

**Issue:** These are optional journal-style choices, not reference-validity defects.

**Why it matters:** Consistent formatting improves readability without changing attribution.

**Recommended fix:** Once the journal is chosen, normalize capitalization, omit a redundant DOI URL if printed twice, and use supported article/page fields. Do not invent Kitsune pages or add a meaningless note solely to silence its harmless warning.

**Exact replacement:** No universal replacement; preserve proper names, acronyms and valid article numbering.

**Change type:** Manuscript only

**Verification:** The chosen style compiles with correct DOI mappings and no fabricated pagination.

**Confidence:** Confirmed

### O2 — Defer venue-specific layout adjustments until venue choice

**ID:** O2 · **Priority:** P3 · **Category:** Presentation

**Location:** main.tex:1,6,13; Table 1; highlights.txt.

**Existing content or behavior:** The draft uses elsarticle preprint, US letter and a custom algorithm counter.

**Evidence:** All 28 pages were inspected; a source-matching temporary rebuild shows clean layout. No venue has been named.

**Issue:** Formatting compatibility cannot be determined before venue selection.

**Why it matters:** Avoids cosmetic work ahead of scientific corrections.

**Recommended fix:** After venue choice, apply its class, algorithm and highlight requirements and recheck floats/citations in a temporary build.

**Exact replacement:** Not applicable until a journal template is chosen.

**Change type:** Manuscript only

**Verification:** Final files follow the chosen venue’s actual instructions.

**Confidence:** Confirmed

### O5 — Offer a zoomed low-FPR view while retaining the complete upper tail

**ID:** O5 · **Priority:** P3 · **Category:** Presentation

**Location:** Figure 3, figures/false-alert-distribution.pdf, PDF p.21.

**Existing content or behavior:** Linear FPR axes extend to roughly .67, compressing the nominal-alpha region.

**Evidence:** The .01/.02 guides occupy only the first few percent of the width; the full tail is accurately visible.

**Issue:** Small FPR differences are difficult to inspect at page scale.

**Why it matters:** A focused view can clarify the tradeoff without hiding worst targets.

**Recommended fix:** Optionally add a low-FPR inset/second panel and retain the full-range plot. If using a log-like axis, show exact zero explicitly rather than silently replacing it with a positive floor.

**Exact replacement:** Caption: “The inset enlarges the low-FPR region; the main panel retains the complete target distribution.”

**Change type:** Code modification (figure generation only, later)

**Verification:** Zero rates and the .67 tail remain represented; nominal guides are readable.

**Confidence:** Confirmed

### O8 — Keep optional citations selective and version-aware

**ID:** O8 · **Priority:** P3 · **Category:** Literature

**Location:** §2.3, §2.4, §5.1 or §8, only where a corresponding point is made.

**Existing content or behavior:** Several sources exist beyond the closest neighbors.

**Evidence:** AnoShift supports shift benchmarking; ADBench has DOI 10.52202/068431-2329 in the official NeurIPS record; Cantone 2024 and Jin/Lee 2026 address transfer limits. Sáez-de-Cámara’s first name is Xabier (not J.), DOI 10.1109/TDSC.2023.3247166. Wardana’s correct authors are Aulia Arif Wardana and Parman Sukarno; its issue is CSUR 57(4), article 88, 2025, online December 2024. Wardana full text was not obtained.

**Issue:** Optional sources must not become padding or carry unverified algorithm claims.

**Why it matters:** A small accurate selection is useful.

**Recommended fix:** Choose only sources needed for the explicit placements in §11. Identify preprint versions separately from final publication metadata.

**Exact replacement:** No mandatory replacement; exact placements and supported claims appear in §11.

**Change type:** Manuscript only

**Verification:** Each citation has a purpose, correct metadata and candid full-text status.

**Confidence:** Confirmed

### W1 — Remove the abstract’s redundant closing generality if space is needed

**ID:** W1 · **Priority:** P3 · **Category:** Writing

**Location:** main.tex:19, abstract final sentence.

**Existing content or behavior:** “The results establish complementary detection benefits in the evaluated benign-only target setting while distinguishing those benefits from operational false-alert guarantees.”

**Evidence:** The preceding sentences already state complementarity and the absence of an FPR bound.

**Issue:** The closing sentence repeats these points. This is an editorial judgment, not authorship evidence.

**Why it matters:** Its space could accommodate the method-specific fusion result.

**Recommended fix:** Delete only if revising the abstract for space; preserve its preceding FPR caveat.

**Exact replacement:** Delete the quoted sentence. Scientific preservation: both points remain explicit above.

**Change type:** Manuscript only

**Verification:** The abstract retains the setting, comparator, main gains and no-future-FPR-guarantee caveat.

**Confidence:** Confirmed

### W2 — Simplify the conclusion’s last sentence

**ID:** W2 · **Priority:** P3 · **Category:** Writing

**Location:** sections/conclusion.tex:4.

**Existing content or behavior:** “PACE therefore provides evidence for complementary peer and local detection under benign-only target adaptation, with independent operational calibration and deployment validation remaining necessary.”

**Evidence:** The contribution and deployment boundary are supported.

**Issue:** Stacked nominal phrases obscure a straightforward conclusion; no scientific error is alleged.

**Why it matters:** A direct final sentence is easier to read.

**Recommended fix:** Replace this sentence only, retaining the supported contribution and boundary.

**Exact replacement:** “The experiments support complementary peer and local detection when the target supplies benign data; deployment still requires independent false-alert validation.” Scientific preservation: contribution and limitation are unchanged.

**Change type:** Manuscript only

**Verification:** The replacement introduces no guarantee, external-superiority claim or field-deployment assertion.

**Confidence:** Confirmed


## 7. Scientific Novelty and Contribution Assessment

The strongest defensible contribution is a specific **calibration and decision architecture for frozen foreign supervised models at a trusted benign-only target**: separate target-relative peer thresholds, a jointly chosen empirical union multiplier, and an independent local anchor/bank route without a final fused threshold. It preserves multiple source models rather than training one target classifier or averaging parameters. This is an integration contribution requiring empirical evidence, not a new base estimator or population-FPR theorem.

Established elements include isolation forest, LOF, kNN, PCA, histograms, regularized Gaussian scores, empirical ranks, Bonferroni/OR allocation, benign-only device modeling, detector-response-space anomaly modeling and hybrid supervised/anomaly evidence. Original ECOD differs from PACE’s explicit featurewise two-tail sum; do not identify the implementation as the original estimator. Bridges provides an ensemble-alert-threshold precedent; CADES provides a valid conformal-score combination precedent. Neither gives PACE a transferred guarantee.

| Neighbor | What overlaps | Meaningful distinction from PACE |
|---|---|---|
| MCDE, Yue et al. 2026 | Supervised plus benign-deviation evidence, empirical tail alignment, branch allocation, low-FPR focus | Mainly family hold-out; one pooled supervised model, benign modes and a final fused threshold; target recalibration in a supplementary study |
| DI-NIDS, Layeghy et al. 2023 | Labeled source knowledge and attack-label-free target transfer | Learns domain-invariant representations using unlabeled targets, followed by a one-class SVM |
| Rey et al. 2022 / DÏoT / FedMSE | Collaborative benign/anomaly modeling and heterogeneous devices | Federated/model aggregation rather than a bank of fixed, separately target-thresholded supervised peers |
| OSDN, Wu et al. 2024 | Source labels and unlabeled target adaptation | Mixed target traffic, heterogeneous projections and alignment; PACE requires trusted benign target roles and compatible schemas |
| Bridges 2017 / CADES 2025 | Per-detector thresholds or complementary-score combination under an alert/FPR budget | Different information regimes and validity assumptions; PACE jointly scales empirical target-block thresholds |
| Xiang et al. 2026 preprint | Trusted-normal target adaptation and temporal empirical calibration | CAN residual detector and block-maxima decision units; PACE uses row decisions and a maximum of block-specific quantiles |
| MEFA, Fiadzeawu et al. 2025 | Multiple source models and target adaptation | Abstract describes representation alignment and pseudo-label-weighted fusion into a target classifier; full methods unavailable, so this comparison remains provisional |

A skeptical reviewer can reasonably call the method a combination of known detectors with empirical budget allocation. The strongest response is **the fixed-component architectural experiment**, M5, together with the complete local-branch and peers-only comparisons, M13. The nominal fusion contrast includes a higher alert rate, so it does not by itself isolate drowning. No measured superiority over external methods is established. The focused search found no verified method with the exact combined PACE information and decision structure; unreviewed or inaccessible literature prevents an exhaustive novelty claim.

Suggested end-of-introduction positioning: “The contribution is a calibration and decision architecture for reusing several frozen supervised peers at a benign-only target, with paired benchmark evidence on separate peer thresholds and complementary local detection.” The current title is defensible; ‘benign-only target data’ must retain the already explicit central-preprocessing and extra-role qualifications. No title change is necessary.

## 8. Mathematical, Code, and Data Verification

The actual pipeline was reconstructed from `core/evidence.py`, `core/decision.py`, `experiment/{inputs,models,arms,runner,splitting}.py`, relevant adapters and `configs/default.yaml`. The manuscript’s equations (1)–(8) and Algorithm 1 agree with the inspected implementation in the evaluated regime.

| Object | Verified implementation and implication |
|---|---|
| Target evidence, Eq.(2) | `searchsorted(..., side="left")` counts reference ties in the upper tail. Beyond the maximum, evidence is log(n+1)+log1p((s−max)/max(max−median,1e−6)). Monotone with a jump at the maximum; a score convention, not a valid p-value. |
| Quantile, Eq.(3) | Index ceil((1−w)(n_j−1)), zero-based; take maximum over blocks; compare strictly `>`. At most floor(w(n_j−1)) strict calibration exceedances per block. |
| Balanced blocks | J=max(1,floor(M/250)); `array_split` gives balanced consecutive blocks. With campaign eligibility, lengths lie in 250–499; J=1 roles can exceed 399 (m15). |
| Union search, Eq.(4) | Equal initial shares; doubling capped at 64 or .999/max(share); 20 bisections retain a feasible lower point. At fixed B the union count is monotone in the multiplier. Equal-share scale 1 is feasible by the union/count bound; arbitrary shares need not be repaired. |
| Resolution, Eq.(7) | The anchor receives B_L when .5 B_L<1/250; the bank retains .5 B_L. Sums .0125 at α=.01 and .05 at α=.05. A heuristic, not a strict partition in all settings. |
| Anchor | Mean of separately reference-transformed PCA, IF, LOF, kNN, histogram and Gaussian scores; no autoencoder and no second transformation. Singleton union calibration, not merely an unscaled quantile. |
| Local bank | Reuses IF/histogram/Gaussian evidence; adds featurewise two-tail ECOD-sum, pair surprisal and IF on K peer evidence coordinates. Reused fitted outputs are not independent copies. |
| Rarity equations (5)–(6) | Featurewise larger negative-log tail before summing; 150 ordered pair draws with replacement, self-pairs removed for d>1; unique 1/8…7/8 quantiles and left insertion; 9×9 allocated tables, smoothing .5. Pair count can be below 150; repeated pairs remain. |
| Final Eq.(8) | OR of peer union, anchor and bank; no final recalibration. Local-exclusive coverage is combined TPR minus the extracted peer branch’s TPR, not the primary comparator gain. |
| Comparator/arms | Seven-member mean-local comparator includes a trained autoencoder; recorded max uses probability evidence; peer-fusion arms use margin evidence and keep PACE’s bank; bank-only receives B_L; local-only mean and peers-only use full α. |
| Local/peer fitting | First N benign training rows fit target detectors. IF/neighbor fit samples capped at 8000; PCA/histogram/Gaussian/rarity use full N. Peer MLP 64–16–2, 8 epochs, balanced cross-entropy; own model excluded. Parameter count 64d+1138 is correct but is not measured resource use. |
| Splits/aggregation | Stable chronological 40/20/20/20 benign roles; per-label attack midpoint and ≤10,000 first rows of each half. Equal attack-type, then seed, then target averaging; UNSW strata share a corpus bootstrap group. |
| Edge conventions | Empty/nonfinite scores are rejected. Ties do not alert. Extrapolation need not be continuous at the largest reference value. No statistical guarantee follows from its name or smoothing. Synthetic arithmetic checks evaluated formulas only; no detector was retrained. |

**Information flow and leakage adjudication:** pooled scaling uses complete benign training roles of the full peer pool, including target rows beyond N, before classifiers are fitted. The target therefore affects peer representations through shared statistics even though no target examples are supplied as that peer’s labeled training rows. This central/transductive preprocessing is accurately disclosed; it is not target-attack-label leakage and it limits a literal model-only deployment interpretation.

Gotham totals are formed before label filtering and sampling, using all packets in the complete second. They can share raw contributors across roles and use later packets. The paper already explicitly discloses this feature dependency; no new claim that labels enter fitting is warranted. Chronology is per client and label, not one prospective network-wide cutoff. Eligibility uses attack availability, so the cohort is a selected benchmark population. Own-model exclusion and separate benign roles were confirmed. Within-client split indices are disjoint; this is not a certification that every corpus contains no duplicate feature vectors, duplicated raw records or related cross-client events. No blanket “no row shared across clients” claim is justified without raw identity checks.

**Affine scaling caution:** rank-only ECOD-sum and quantile-cell pair counts are invariant under exact positive affine feature maps. The implemented system is not universally invariant. `evidence.py` fixes a 1e−6 spread floor: reference [0,0,0], query 1e−7 produces 1.481605; multiplying both by 10 gives 2.079442. Histograms pad a training-constant feature by ±.5 regardless of scale: 100 zero training values with queries .01/.04 have score .172371, whereas scaling by 10 gives 4.787492. Finite-precision tree splitting is another reason not to assert blanket implemented invariance. These facts do not contradict the current manuscript, which already discloses preprocessing; no added universal-invariance sentence is recommended.

## 9. Experimental and Statistical Assessment

An independent JSON loader used the full unique key (α,N,stratum,target,seed), confirmed 408 identical repeats, and reconstructed seed-averaged target metrics. A separately written NumPy bootstrap follows the documented corpus order, lexical target order, 10,000 replicates, seed 314159 and linear endpoint quantiles. It exactly reproduces the primary intervals to six decimals. A further 4,528 scalar checks across target, fixed-cohort scaling, seed, FPR and decomposition tables found no discrepancies at their printed precision. This is verification of calculations on frozen evidence, not a new experimental campaign.

| α | N | PACE TPR | Mean-local comparator TPR | Difference [95% descriptive interval] | PACE/comparator FPR |
|---|---:|---:|---:|---|---|
| .01 | 100 | .833405 | .807318 | .026086 [.011323,.043346] | .033992 / .030516 |
| .01 | 400 | .838987 | .810681 | .028306 [.012339,.047375] | .032369 / .030781 |
| .05 | 100 | .894767 | .871348 | .023418 [.005777,.043968] | .055453 / .057437 |
| .05 | 400 | .896837 | .873619 | .023219 [.003116,.045968] | .054337 / .064407 |

The complete internal arm inventory below uses four cells ordered **(.01,100),(.01,400),(.05,100),(.05,400)**. Each cell is **arm TPR / mean FPR**; the next table gives PACE-minus-arm effects and intervals. All are nominal comparisons.


| Rule | .01 / 100 | .01 / 400 | .05 / 100 | .05 / 400 |
|---|---|---|---|---|
| Shared-peer mean local | 0.807318 / 0.030516 | 0.810681 / 0.030781 | 0.871348 / 0.057437 | 0.873619 / 0.064407 |
| Without feature family | 0.829992 / 0.031778 | 0.836974 / 0.031273 | 0.894640 / 0.052064 | 0.896493 / 0.053259 |
| Without response family | 0.831397 / 0.033859 | 0.836191 / 0.032074 | 0.891649 / 0.055375 | 0.895568 / 0.054558 |
| Without rarity family | 0.824778 / 0.033800 | 0.827010 / 0.032295 | 0.888746 / 0.055281 | 0.886434 / 0.053995 |
| Bank only at B_L | 0.823313 / 0.033096 | 0.824088 / 0.031255 | 0.897598 / 0.060458 | 0.898372 / 0.064650 |
| Single local union | 0.833405 / 0.033998 | 0.838987 / 0.032371 | 0.892561 / 0.057816 | 0.897186 / 0.065082 |
| Response on margins | 0.832304 / 0.033951 | 0.837281 / 0.032211 | 0.891888 / 0.056676 | 0.895565 / 0.058334 |
| Reserve .25 | 0.814000 / 0.033265 | 0.818840 / 0.031290 | 0.897066 / 0.060030 | 0.901835 / 0.060317 |
| Mean-fused peers + PACE local | 0.759968 / 0.026093 | 0.783009 / 0.024896 | 0.842308 / 0.047157 | 0.852114 / 0.044361 |
| Max-fused peers + PACE local | 0.787850 / 0.025120 | 0.802597 / 0.023855 | 0.864283 / 0.046431 | 0.865193 / 0.043049 |
| Peers only at full α | 0.812411 / 0.030775 | 0.812411 / 0.030775 | 0.883945 / 0.066227 | 0.883945 / 0.066227 |
| Seven-detector mean only at full α | 0.388775 / 0.019588 | 0.475443 / 0.021635 | 0.561134 / 0.050272 | 0.628398 / 0.055778 |
| Recorded max-fusion + mean local | .742730 / .021193 | .760522 / .022415 | .815769 / .044953 | .822336 / .052848 |
| Recorded Bonferroni + mean local | .795154 / .029332 | .798159 / .029579 | .832897 / .046402 | .839911 / .051129 |

Paired nominal detection effects:

| PACE minus arm | .01 / 100 | .01 / 400 | .05 / 100 | .05 / 400 |
|---|---|---|---|---|
| Shared-peer mean local | 0.026086 [0.011323,0.043346] | 0.028306 [0.012339,0.047375] | 0.023418 [0.005777,0.043968] | 0.023219 [0.003116,0.045968] |
| Without feature family | 0.003413 [0.000785,0.007666] | 0.002013 [0.000496,0.003905] | 0.000127 [-0.002145,0.002024] | 0.000344 [-0.000063,0.000897] |
| Without response family | 0.002008 [0.000277,0.004518] | 0.002796 [0.000253,0.006454] | 0.003118 [-0.000083,0.008302] | 0.001269 [0.000021,0.003571] |
| Without rarity family | 0.008626 [0.000602,0.020504] | 0.011977 [0.002783,0.023672] | 0.006020 [0.000008,0.015617] | 0.010403 [0.000237,0.023050] |
| Bank only at B_L | 0.010092 [-0.017927,0.052456] | 0.014899 [-0.009737,0.054637] | -0.002832 [-0.005608,-0.000786] | -0.001535 [-0.003242,-0.000329] |
| Single local union | 0.000000 [0.000000,0.000000] | 0.000000 [0.000000,0.000000] | 0.002206 [-0.000224,0.006605] | -0.000349 [-0.000868,0.000045] |
| Response on margins | 0.001101 [-0.001171,0.003902] | 0.001706 [-0.001722,0.005636] | 0.002879 [-0.000354,0.008044] | 0.001272 [0.000021,0.003573] |
| Reserve .25 | 0.019404 [0.000473,0.055006] | 0.020147 [0.001275,0.055773] | -0.002299 [-0.006719,0.003283] | -0.004998 [-0.008622,-0.001879] |
| Mean-fused peers + PACE local | 0.073437 [0.029755,0.119872] | 0.055978 [0.015296,0.098622] | 0.052459 [0.021191,0.091373] | 0.044723 [0.017602,0.080736] |
| Max-fused peers + PACE local | 0.045555 [0.008586,0.083883] | 0.036390 [0.001197,0.071494] | 0.030484 [0.014488,0.048558] | 0.031644 [0.016224,0.049405] |
| Peers only at full α | 0.020993 [-0.011591,0.056029] | 0.026576 [-0.007605,0.062007] | 0.010821 [-0.008014,0.032468] | 0.012892 [-0.007048,0.036236] |
| Seven-detector mean only at full α | 0.444630 [0.352850,0.532565] | 0.363544 [0.278002,0.448569] | 0.333633 [0.249867,0.419775] | 0.268439 [0.189487,0.351664] |


**Interpretation:** M13 addresses omitted rules; M5 covers fused peers; M20 covers bank-only reallocation. The fixed-share Bonferroni comparison holds the seven-member local branch fixed: comparator-minus-Bonferroni is .012165,.012522,.038451,.033707, with FPR costs .001184,.001202,.011036,.013279. It does not establish matched-FPR efficiency of union scaling.

The target-macro estimand is appropriate for heterogeneous capture sizes, but it weights corpora by target count, not equally. An alternative equal-corpus mean would be a different exploratory estimand, not a correction to the primary one. Three seeds document stability under specified learning randomness on these roles; they do not establish that three is universally sufficient, nor estimate unseen-target or unseen-corpus variation. Shared models/captures invalidate a naive independence interpretation for either a bootstrap or sign test. More seeds can improve precision for randomness-sensitive targets; they cannot remove development-target reuse. Multiple ablations, headroom splits and deletion sensitivities are exploratory; no simultaneous family-wise coverage is claimed.

**Budget behavior:** fixed-cohort N=100→400 effects are .005582 [−.003760,.015387] at α=.01 and .002070 [−.004456,.008690] at α=.05. The 15-target N=100→20,000 effects and their dominant host are in M14. The .05/N=2000 ex-TON negative gain and uncertain adjusted larger-budget results must remain. Resource cohorts are never compared as if their differing target composition were a budget effect.

**Complementarity:** local-exclusive fractions .147434 (TON_IoT), .120957 (IoT-23), .000042 (N-BaIoT) at .01/N=400 reproduce via union identities. They are neither local-only-arm rates nor gains over the primary comparator. Family-removal results change recalibration and are not additive expert importance scores.

**False alerts:** Table 5’s means, medians, 90th percentiles, maxima and exceedance shares reproduce. Medians below .01 coexist with means above .03 and worst targets near .67. The same extreme target under several methods is relevant diagnostic context, not a reason to dismiss deployment risk. Benchmark class sampling provides no incident precision, alerts per hour or operational prevalence estimate.

**Engineering acceptance:** the recorded primary verdict follows internal tolerances, not journal acceptance or formal statistical non-inferiority. The .01/N=100 mean-FPR increase .003476 is close to its .004 tolerance; target–seed exceedance increase .028736 is close to .03. A positive clamped-interval gate inherits M12’s interpretation. Freezing criteria before the confirmation outcomes is an author-verification issue (M16), independent of whether the arithmetic passes.


## 10. Reference Verification

The 29 cited entries were independently matched against Crossref (27) and DataCite (2), including title/subtitle, author list, publication year, venue and identifier. Final-publication metadata is distinguished from preprint headers. Across the cited, recommended/candidate and two contextual-preprint inventories, 50 bibliographic records were independently identified: 48 verified DOI records (43 Crossref, five DataCite) and two authoritative non-DOI proceedings records. This counts identity/metadata checks, not 50 full-paper readings; the supporting thesis is listed separately. All 29 keys occur in the compiled bibliography/citations. The matrix retains citation order so existing `Ref_NN` filenames remain stable. “Supported” means the relevant method/claim passages were inspected in available text; it does not assert that every page of every article was read. Four cited article PDFs remain unavailable. Xiang’s full HTML is now accessible and supports the manuscript’s block-maxima/dependence wording; it should not be downgraded to abstract-only verification.

| # | Reference | DOI / ID | Metadata | Local PDF (version) | Where cited | Claim support |
|---|---|---|---|---|---|---|
| 1 | Zimek, Campello, Sander 2014, SIGKDD Explor. 15(1):11–22 | [10.1145/2594473.2594476](https://doi.org/10.1145/2594473.2594476) | Correct | `Ref_01_Zimek_2014.pdf` | §1, §2.5 | Supported ("overestimate the outlierness") |
| 2 | Yue, Lu, Xie 2026, Entropy 28(9):1026 | [10.3390/e28091026](https://doi.org/10.3390/e28091026) | Correct | `Ref_02_Yue_2026.pdf` (version of record) | §1, §2.1, Table 1, §8 | Supported (HGB, benign modes, tail evidence, branch allocation, family hold-out, trusted-benign recalibration study). Also evaluates CICIoT2023. |
| 3 | Mirsky et al. 2018, NDSS (Kitsune) | [10.14722/ndss.2018.23204](https://doi.org/10.14722/ndss.2018.23204) | Correct; no pages (NDSS) | `Ref_03_Mirsky_2018.pdf` (arXiv) | §2.5, Table 1 | Supported |
| 4 | Nguyen et al. 2019, ICDCS (DÏoT), 756–767 | [10.1109/ICDCS.2019.00080](https://doi.org/10.1109/ICDCS.2019.00080) | Correct (pages present) | `Ref_04_Nguyen_2019.pdf` (accepted manuscript) | §2.4, Table 1 | Supported (window anomaly-triggering condition) |
| 5 | Wu et al. 2024, ACM TOIT 24(1):1–26 (OSDN) | [10.1145/3639822](https://doi.org/10.1145/3639822) | Correct | `Ref_05_Wu_2024.pdf` (arXiv v2) | §2.4, Table 1 | Supported |
| 6 | Angelopoulos & Bates 2023, FnT ML 16(4):494–591 | [10.1561/2200000101](https://doi.org/10.1561/2200000101) | Correct | `Ref_06_Angelopoulos_2023.pdf` (arXiv "A Gentle Introduction to Conformal Prediction and Distribution-Free UQ"; differs from the FnT title) | §2.2 | Supported |
| 7 | Bates et al. 2023, Ann. Stat. 51(1):149–178 | [10.1214/22-AOS2244](https://doi.org/10.1214/22-AOS2244) | Correct | `Ref_07_Bates_2023.pdf` (arXiv v3) | §2.2 | Supported (marginal vs calibration-conditional validity) |
| 8 | Vovk & Wang 2020, Biometrika 107(4):791–808 | [10.1093/biomet/asaa027](https://doi.org/10.1093/biomet/asaa027) | Correct | `Ref_08_Vovk_2020.pdf` (author manuscript) | §2.2 | Supported |
| 9 | Barber et al. 2023, Ann. Stat. 51(2):816–845 | [10.1214/23-AOS2276](https://doi.org/10.1214/23-AOS2276) | Correct | `Ref_09_Barber_2023.pdf` (arXiv v5) | §2.2 | Supported (weighted quantiles, coverage gap) |
| 10 | Xiang, Yang, So 2026, Preprints.org | [10.20944/preprints202609.1557.v1](https://doi.org/10.20944/preprints202609.1557.v1) | Correct; preprint | PDF download blocked (403); full publisher HTML read | §2.2 | Supported: §2.3 explicitly defines block maxima, dependence-informed length and resolution constraint; no wording correction needed |
| 11 | Liu, Ting, Zhou 2008, ICDM 413–422 | [10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17) | Correct | `Ref_11_Liu_2008.pdf` | §2.3, §4.4 | Supported |
| 12 | Breunig et al. 2000, SIGMOD 93–104 | [10.1145/342009.335388](https://doi.org/10.1145/342009.335388) | Correct | Not obtained (ACM paywall; LMU mirror returns an HTML page) | §2.3, §4.4 | Used only to attribute LOF ("compares neighborhood density with that of nearby observations"); matches the well-known definition; low risk |
| 13 | Pang et al. 2021, ACM CSUR 54(2) art. 38 | [10.1145/3439950](https://doi.org/10.1145/3439950) | Correct | `Ref_13_Pang_2021.pdf` (arXiv v3) | §2.3 | Supported for the narrow claim |
| 14 | Meidan et al. 2018, IEEE Pervasive Comput. 17(3):12–22 | [10.1109/MPRV.2018.03367731](https://doi.org/10.1109/MPRV.2018.03367731) | Correct | `Ref_14_Meidan_2018.pdf` (arXiv; stale header) | §2.4, §5.1 | Supported (tr and ws optimized on benign DS_opt). Add dataset DOI [10.24432/C5RC8J](https://doi.org/10.24432/C5RC8J) (m17). |
| 15 | Nguyen & Beuran 2025, Comput. Secur. 151:104337 | [10.1016/j.cose.2025.104337](https://doi.org/10.1016/j.cose.2025.104337) | Correct | `Ref_15_Nguyen_2025.pdf` (author manuscript) | §2.4 | Supported |
| 16 | Abhijit et al. 2025, Sci. Rep. 15:33974 | [10.1038/s41598-025-11659-7](https://doi.org/10.1038/s41598-025-11659-7) | Correct | `Ref_16_Abhijit_2025.pdf` (version of record) | §2.4 | Supported |
| 17 | Talpini et al. 2025, ICAART 840–848 | [10.5220/0013376600003890](https://doi.org/10.5220/0013376600003890) | Correct | `Ref_17_Talpini_2025.pdf` | §2.4 | Supported (federated Platt scaling) |
| 18 | Wang et al. 2022, CVPR 4911–4920 (ViM) | [10.1109/CVPR52688.2022.00487](https://doi.org/10.1109/CVPR52688.2022.00487) | Correct (pages present) | `Ref_18_Wang_2022.pdf` (arXiv) | §2.5 | Supported |
| 19 | Jin 2026, Sensors 26(4):1184 | [10.3390/s26041184](https://doi.org/10.3390/s26041184) | Correct | `Ref_19_Jin_2026.pdf` (version of record) | §2.6 | Supported (random-window vs capture-wise; transferred thresholds) |
| 20 | Ha & Kim 2026, Appl. Sci. 16(5):2284 | [10.3390/app16052284](https://doi.org/10.3390/app16052284) | Correct | `Ref_20_Ha_2026.pdf` (version of record) | §2.6 | Supported |
| 21 | Li et al. 2023, IEEE TKDE 35(12):12181–12193 (ECOD) | [10.1109/TKDE.2022.3159580](https://doi.org/10.1109/TKDE.2022.3159580) | Correct (key says 2022; printed year 2023) | `Ref_21_Li_2023.pdf` (arXiv v3) | §4.4 | Supported (skewness-selected tail) |
| 22 | Garcia et al. 2020, IoT-23, Zenodo | [10.5281/zenodo.4743746](https://doi.org/10.5281/zenodo.4743746) | Correct (v1.0.0, CC BY 4.0) | Dataset | §5.1 | Dataset identity valid; exact small-archive provenance should be added (m17) |
| 23 | Moustafa 2021, Sustain. Cities Soc. 72:102994 | [10.1016/j.scs.2021.102994](https://doi.org/10.1016/j.scs.2021.102994) | Correct | Not obtained (no open copy; Elsevier only) | §5.1 | Corpus attribution only; dataset files used are the TON_IoT "Processed_Network_dataset" release; low risk |
| 24 | Belarbi et al. 2025, Gotham dataset, Zenodo | [10.5281/zenodo.14502760](https://doi.org/10.5281/zenodo.14502760) | Correct (CC BY 4.0) | Dataset | §5.1 | Correct; add the 2026 data descriptor (m17) |
| 25 | García et al. 2014, Comput. Secur. 45:100–123 | [10.1016/j.cose.2014.05.011](https://doi.org/10.1016/j.cose.2014.05.011) | Correct | Not obtained (Stratosphere page links only to ScienceDirect) | §5.1 | Corpus attribution only (CTU-13 scenarios, Background discarded); low risk |
| 26 | Hamza et al. 2019, SOSR 36–48 | [10.1145/3314148.3314352](https://doi.org/10.1145/3314148.3314352) | Correct | `Ref_26_Hamza_2019.pdf` | §5.1 | Supported (minute flow counters; dataset released) |
| 27 | Sommer & Paxson 2010, IEEE S&P 305–316 | [10.1109/SP.2010.25](https://doi.org/10.1109/SP.2010.25) | Correct | `Ref_27_Sommer_2010.pdf` | §7.4 | Supported |
| 28 | Saito & Rehmsmeier 2015, PLOS ONE 10(3):e0118432 | [10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432) | Correct | `Ref_28_Saito_2015.pdf` | §7.4 | Supported |
| 29 | Arp et al. 2024, CACM 67(11):104–112 | [10.1145/3643456](https://doi.org/10.1145/3643456) | Correct | `Ref_29_Arp_2024.pdf` | §8 | Supported |



**Access attempts:** Ref 10 PDF 403; Ref 12 author PDF redirects to HTML or unavailable, with indexed primary text only; Ref 23 Elsevier API 429; Ref 25 Elsevier API 406; R4 institutional PDF 404 (supporting thesis obtained); R7/R15 ACM PDF 403; MEFA Wiley PDF 403. No paywall or authentication was bypassed.

**Access and version limitations:** Ref 10 is an explicitly labeled preprint, verified in full HTML at [Preprints.org](https://www.preprints.org/manuscript/202609.1557), §§2.3 and 3.2; its PDF remains blocked. Refs 12,23,25 have valid metadata, but this audit did not obtain their full PDFs. LOF’s narrow definition is supported by indexed primary excerpts, not a fresh complete-paper reading. TON_IoT and CTU-13 are used for corpus attribution; their cache/input schemas were checked separately. No unresolved full-text access is silently treated as successful verification. The matrix’s “low risk” language for these attribution-only uses is an assessment of the attached claim, not a claim to have read inaccessible papers.

Reference PDF integrity was checked by PDF header, successful Poppler parsing/page count, extracted text and matching title/author cues. The 36 existing files were reused. `Proposed_05_Rey_2022.pdf` is an older preprint with an additional author compared with the final DOI record; `Proposed_13_Cantone_2024.pdf` has a different preprint title; `Ref_06_Angelopoulos_2023.pdf` has the longer arXiv title; `Ref_14_Meidan_2018.pdf` has a stale journal header. These are matching author versions, not wrong-file mappings. No uncertain file was deleted or renamed.

**Corrections and retractions:** metadata relations and 42 successful Crossref update-notice queries returned no notices in the queried records. This is not proof of no corrections everywhere. The official [USENIX TESSERACT erratum](https://www.usenix.org/system/files/sec19_errata_6.pdf) corrects an extra normalization factor in Eq.(4) and states that computed results remain valid. This non-DOI proceedings correction is outside a Crossref-only check. It does not change the temporal/spatial-bias citation purpose. Publisher-version and independent retraction coverage is not exhaustive; check relevant Crossmark/publisher records again before submission.

## 11. Recent Literature and Recommended Citations

A fresh search on 20 October 2026 used multiple web formulations, two SciSpace semantic searches, Crossref/DataCite metadata, publisher pages and scholarly/author repositories. Queries covered benign-only/cold-start transfer; source-free and multi-source IDS; frozen source classifiers; collaborative/federated/personalized detectors; open-set detection; conformal/low-FPR calibration; detector unions; distribution shift; and benchmark quality. Results available after 20 October were not used. Search-index dates were checked against primary records; indexed dates or abstracts alone were not accepted as proof of peer review or full-method verification. This is a focused substantive search, not an exhaustive systematic review.

The following inventory lists recommended and optional sources with their actual access status. Priorities refer to citation utility, not source prestige. Add only the sources needed for explicit claims.


| ID | Citation | DOI / ID | Type | Local file | Placement and reason | Effect | Priority |
|---|---|---|---|---|---|---|---|
| R1 | G. Apruzzese, "SoK: Reshaping Research on Network Intrusion Detection Systems," ACM AsiaCCS 2026, 1028–1042 | [10.1145/3779208.3807488](https://doi.org/10.1145/3779208.3807488) | Peer-reviewed conference | `Proposed_03_Apruzzese_2026.pdf` (arXiv 2604.17556 v1) | §7.4 after "An anomalous record need not represent an actionable security event": operators triage reports, not single samples | Supports the row-alert vs incident caveat | P2 |
| R2 | G. Apruzzese, P. Laskov, J. Schneider, "SoK: Pragmatic Assessment of Machine Learning for Network Intrusion Detection," IEEE EuroS&P 2023, 592–614 | [10.1109/EuroSP57164.2023.00042](https://doi.org/10.1109/EuroSP57164.2023.00042) | Peer-reviewed conference | `Proposed_02_Apruzzese_2023.pdf` (arXiv) | §8 beside Arp et al. | Evaluation framing | P2 |
| R3 | Feargus Pendlebury, Fabio Pierazzi, Roberto Jordaney, Johannes Kinder, Lorenzo Cavallaro, "TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time," USENIX Security 2019 | No DOI (USENIX); arXiv 1807.07838 | Peer-reviewed conference | `Proposed_01_Pendlebury_2019.pdf` | §8, temporal and spatial bias | Supports chronological roles and their limits | P2 |
| R4 | R. Flood, G. Engelen, D. Aspinall, L. Desmet, "Bad Design Smells in Benchmark NIDS Datasets," IEEE EuroS&P 2024, 658–675 | [10.1109/EuroSP60621.2024.00042](https://doi.org/10.1109/EuroSP60621.2024.00042) | Peer-reviewed conference | Paper PDF not obtained (IEEE; no repository copy). Content read in R. Flood's open Edinburgh PhD thesis (era.ed.ac.uk handle 1842/43694, chapters 3–4), which reproduces the study; stored separately as `Supporting_Flood_2025_Thesis.pdf`; not represented as the conference PDF | §8 corpus paragraph and §5.1 (M1 item 5): the audited seven datasets include TON_IoT and CTU-13; TON_IoT labels background DNS as malicious and launches backdoor attacks against closed ports; CTU-13 has granular labels but some mislabelled flows | Context for corpus-specific behavior; PACE subset impact is not established from a destination-port proxy alone; do not claim complete absence of label artefacts (M1) | P2 |
| R5 | V. Rey et al., "Federated learning for malware detection in IoT devices," Computer Networks 204 (2022) 108693 | [10.1016/j.comnet.2021.108693](https://doi.org/10.1016/j.comnet.2021.108693) | Journal | `Proposed_05_Rey_2022.pdf` (arXiv 2104.09994; preprint includes an additional author) | §2.4: "Rey et al. train federated benign-only autoencoders and supervised classifiers on N-BaIoT, including devices unseen in training; PACE instead freezes per-peer supervised models and calibrates them separately on the target." | Prior art on PACE's own corpus | P2 |
| R6 | S. Layeghy, M. Baktashmotlagh, M. Portmann, "DI-NIDS: Domain invariant network intrusion detection system," Knowledge-Based Systems 273 (2023) 110626 | [10.1016/j.knosys.2023.110626](https://doi.org/10.1016/j.knosys.2023.110626) | Journal | `Proposed_04_Layeghy_2023.pdf` (arXiv 2210.08252) | §2.4 after OSDN: labeled sources plus unlabeled target, domain-invariant features, one-class SVM | Closest cross-domain benign-anomaly neighbor | P2 |
| R8 | O. Belarbi, T. Spyridopoulos, E. Anthi, O. Rana, P. Carnelli, A. Khan, "A Device-Level IoT Network Traffic Dataset with Distributed Capture and Non-IID Characteristics," Data 11(8) (2026) 207 | [10.3390/data11080207](https://doi.org/10.3390/data11080207) | Journal (data descriptor) | `Proposed_06_Belarbi_2026.pdf` (version of record; read) | §5.1 next to Ref 24 | Peer-reviewed description of the Gotham dataset used | P2 |
| R9 | R. A. Bridges, J. D. Jamieson, J. W. Reed, "Setting the threshold for high throughput detectors: A mathematical approach for ensembles of dynamic, heterogeneous, probabilistic anomaly detectors," IEEE BigData 2017, 1071–1078 | [10.1109/BigData.2017.8258031](https://doi.org/10.1109/BigData.2017.8258031) | Peer-reviewed conference | `Proposed_07_Bridges_2017.pdf` (arXiv 1710.09422; read) | §2.2 or §2.5 (O7) | Precedent for per-detector thresholds under a joint alert budget; sharpens novelty claim | P2 |
| R10 | S. Zhang, C. Zhou, Y. Liu, P. Zhang, X. Lin, S. Pan, "Conformal Anomaly Detection in Event Sequences," ICML 2025, PMLR 267:77177–77193 | No publication DOI verified; authoritative PMLR archival record | Peer-reviewed conference | `Proposed_08_Zhang_2025.pdf` (PMLR; read) | §2.2 (O7) | Bonferroni-combined conformal scores with finite-sample FPR control: the valid-p-value alternative to PACE's empirical union | P2 |
| R11 | N-BaIoT dataset, UCI Machine Learning Repository (Meidan, Bohadana), 2018 | [10.24432/C5RC8J](https://doi.org/10.24432/C5RC8J) | Dataset | — | §5.1 next to Ref 14 | Dataset citation (m17) | P2 |
| R7 | Emmanouil Vasilomanolakis, Shankar Karuppayah, Max Mühlhäuser, Mathias Fischer, "Taxonomy and Survey of Collaborative Intrusion Detection," ACM CSUR 47(4) (2015) 1–33 | [10.1145/2716260](https://doi.org/10.1145/2716260) | Journal (foundational) | Not downloaded (metadata verified) | §1 or §2, one clause | Terminology | P3 |
| R12 | Marius Dragoi, Elena Burceanu, Emanuela Haller, Andrei Manolache, Florin Brad, "AnoShift: A Distribution Shift Benchmark for Unsupervised Anomaly Detection," NeurIPS 2022 Datasets and Benchmarks, 32854–32867 | [10.52202/068431-2381](https://doi.org/10.52202/068431-2381) | Peer-reviewed conference | `Proposed_11_Dragoi_2022.pdf` (arXiv) | §6.4 or §8: benign traffic drift degrades unsupervised detectors over time | Supports M17 framing | P3 |
| R13 | Songqiao Han, Xiyang Hu, Hailiang Huang, Minqi Jiang, Yue Zhao, "ADBench: Anomaly Detection Benchmark," NeurIPS 2022 Datasets and Benchmarks | [10.52202/068431-2329](https://doi.org/10.52202/068431-2329) (official NeurIPS page and Crossref verified) | Peer-reviewed conference | `Proposed_12_Han_2022.pdf` | §2.3: "none of the unsupervised algorithms is statistically better than others" | Motivates the heterogeneous bank | P3 |
| R14 | H. Jin, H. Lee, "Benchmark-Shift-Aware Intrusion Detection for Evolving Network Traffic: Cross-Dataset Generalization, Calibrated Alerting, and Score-Orientation Diagnostics," Electronics 15(17) (2026) 3761 | [10.3390/electronics15173761](https://doi.org/10.3390/electronics15173761) | Journal | `Proposed_10_Jin_2026.pdf` (version of record; read) | §2.6: chronological validation-derived thresholds and nominal alert budgets do not transfer | Supports FPR caveats; same first author as Ref 19 — cite at most one unless both are needed | P3 |
| R15 | A. A. Wardana, P. Sukarno, "Taxonomy and Survey of Collaborative Intrusion Detection System using Federated Learning," ACM CSUR 57(4), article 88 (2025), 1–36; online Dec. 2024 | [10.1145/3701724](https://doi.org/10.1145/3701724) | Journal (metadata/abstract verified; full text unavailable) | Not downloaded (ACM automated download blocked); metadata verified | §2.4 | Modern collaborative-IDS survey; could replace R7 | P3 |
| R16 | Marco Cantone, Claudio Marrocco, Alessandro Bria, "Machine Learning in Network Intrusion Detection: A Cross-Dataset Generalization Study," IEEE Access 12 (2024) 144489–144508 | [10.1109/ACCESS.2024.3472907](https://doi.org/10.1109/ACCESS.2024.3472907) | Journal | `Proposed_13_Cantone_2024.pdf` (arXiv 2402.10974 v1) | §8: why transfer is restricted within strata | Optional context | P3 |
| R17 | Xabier Sáez-de-Cámara, Jose Luis Flores, Cristóbal Arellano, Aitor Urbieta, Urko Zurutuza, "Gotham Testbed: A Reproducible IoT Testbed for Security Experiments and Dataset Generation," IEEE TDSC 21(1) (2024) 186–203 | [10.1109/TDSC.2023.3247166](https://doi.org/10.1109/TDSC.2023.3247166) | Journal | `Proposed_09_SaezDeCamara_2024.pdf` (arXiv v3) | §5.1, one clause | Testbed provenance | P3 |
| R18 | Muhammad Nouman, Muhsin Hassanu, Raja Ujjan, “Calibration Granularity, Not Contamination: Diagnosing a TCN Anomaly Detector’s False Positive Advantage in Cross-Dataset IoT Traffic,” Future Internet 18(9):447 (2026) | [10.3390/fi18090447](https://doi.org/10.3390/fi18090447) | Published peer-reviewed journal article, 24 August 2026 | `Proposed_14_Nouman_2026.pdf`, 45-page version of record; §§3.7,4.7,5.2 read | §2.6, one sentence distinguishing scoring architecture from calibration granularity (O9) | Supports careful calibration diagnosis, not PACE’s FPR causality or performance | P2 |

**Recommended selection:** highest-value additions are R1 (2026 operational framing), R4 (2024 dataset methodology, with thesis-access note), R8 (2026 actual Gotham descriptor), R9 (ensemble-threshold prior art), and R5/R6 (close collaborative/transfer neighbors). R18 is a useful new 2026 example if it replaces generic calibration discussion. R14/R16/R17 are optional precise context. This gives seven missing 2024–2026 **DOI-verified** research options (R1,R4,R8,R14,R16,R17,R18); do not add all seven by default. CADES (R10) is an additional important verified peer-reviewed proceedings neighbor, but no publication DOI was verified: retain its exact [PMLR record](https://proceedings.mlr.press/v267/zhang25dn.html), rather than inventing one or counting it as DOI-verified. TESSERACT likewise has a verified archival USENIX record, not an invented DOI.

**Fresh candidate requiring full-text follow-up, not a fully verified citation recommendation:** Daniel Perez Fiadzeawu, Jielun Zhang and Fuhao Li, “MEFA: Multisource Entropy-Weighted Feature Adaptation for Cross-Domain Intrusion Detection,” Security and Privacy 8(5):e70069 (2025), [10.1002/spy2.70069](https://doi.org/10.1002/spy2.70069). Publisher/Crossref metadata and abstract were checked; PDF/full-text access returns 403 or the abstract page. The abstract describes alignment and confidence-weighted source pseudo-label fusion into a target classifier. It is a meaningful multi-source neighbor to obtain before a strong exhaustive novelty assertion, but PACE’s exact nearest-method comparison cannot be certified from that abstract. Wardana (R15) and Vasilomanolakis (R7) also remain full-text follow-ups rather than essential additions.

**Version-aware preprint context:**

- Michel A. Youssef, *Stream Assembly Is an Uncontrolled Treatment in Streaming Intrusion-Detection Benchmarks*, arXiv:2605.24696v3, 14 September 2026, [10.48550/arXiv.2605.24696](https://doi.org/10.48550/arXiv.2605.24696), `Context_Youssef_2026_v3.pdf`. The current primary arXiv record explicitly replaces the earlier CALIBURN title and identifies substantial corrections. Relevant stream-assembly and score-composition discussion was inspected. Do not retain an obsolete CALIBURN description as verified conformal-risk-control prior art. This is an unreviewed methodological context source, not a PACE baseline or theorem to import.
- Zhenpeng Li, *Robust Conformal Intrusion Detection via Traffic-Aware Calibration and Attack-Orbit Invariance*, arXiv:2609.19241v1, primary record dated 22 July 2026, [10.48550/arXiv.2609.19241](https://doi.org/10.48550/arXiv.2609.19241), `Context_Li_2026_v1.pdf`. DataCite verifies the identifier despite an arXiv page label saying registration pending. Its scope is adversarial perturbation and representation invariance for LLM-based IDS, not benign-only peer transfer. Relevant calibration/assumption passages were read; no published acceptance was verified. Optional security-threat context only.

Other fresh hits, including Choir-IDS, BAROP and a NOMS 2026 rare-attack FL study, were screened through publisher or author-institution records and were not added as fully checked methods: they concern federated model updating, probability calibration, adversarial participation or rare-class partitioning, rather than the exact fixed-peer target-calibration rule. Their full algorithms were not verified and no superiority or absence-of-overlap claim is based on them. The precise incomplete search boundary is preferable to declaring all 2025–2026 work weak or irrelevant.

**Supporting full-text source:** Robert Flood, *Enhancing Trust in NIDS: From Flawed Benchmarks to Formal Guarantees*, PhD thesis, University of Edinburgh, 2025, [institutional handle 1842/43694](https://era.ed.ac.uk/handle/1842/43694). Chapters 3–4 reproduce the dataset-design work and discuss TON_IoT DNS/background labels and failed attack generation. Stored as `Supporting_Flood_2025_Thesis.pdf` for personal research under the repository’s terms. It is not mislabeled as the 2024 conference article. PACE’s destination-specific retained attack sets have very little destination-port-53 traffic in the previously examined diagnostics, but that proxy cannot certify correctness of all labels or absence of background DNS; do not use it to state “known labelling artefacts do not enter the evaluated sets”.


## 12. Originality and Plagiarism Review

A fresh source comparison tokenized the compiled manuscript body before the references, normalized case/ligatures, and compared exact six- and eight-word sequences against the locally available research PDFs. The inspected matches were generic language, the correctly cited MCDE method name and “labeled source data and unlabeled target data” in attributed domain-adaptation context. No eight-word body match or actionable close-paraphrase concern was identified by this limited method. Numeric axis sequences are not prose evidence.

**Confirmed problematic overlap: 0. Suspected actionable textual-overlap concerns: 0.** This is not an institutional similarity report, a percentage or proof of originality. The inaccessible complete texts were not compared; paraphrase beyond exact n-grams and unindexed sources are not exhaustively covered. Author identities and publication histories remain pending, so self-overlap, duplicate submission and relationships to cited work require author confirmation under C1. Manuscript figures have a common project style, but missing source generators (m13) prevent a provenance certification; no externally copied figure was established.

## 13. AI-Style and Wording Improvements

No reliable AI-authorship percentage can be inferred from prose, and none is reported. Formal technical language and necessary caveats should remain. Four narrow wording opportunities are actionable; all preserve the supported scientific meaning. Exact originals below were checked in the current sources.

| Location | Original text | Specific weakness | Exact remedy / preservation |
|---|---|---|---|
| Introduction paragraph 3 | “PACE, Peer-wise Calibrated Experts, maps each supervised peer’s margin relative to scores on a target benign reference role.” | Unclear first description of the transform | W7 gives a replacement including upper-tail ranks **and** extrapolation; no estimator change |
| Discussion §7.2 final sentence | “These results support a detection advantage at the tested nominal parameters, accompanied by an alert-rate tradeoff.” | Too generic where the specific nominal tradeoff should be named | M5 supplies the replacement naming higher mean FPR and approximate sweep sensitivity; no exact-FPR superiority added |
| Abstract final sentence | “The results establish complementary detection benefits in the evaluated benign-only target setting while distinguishing those benefits from operational false-alert guarantees.” | Repeats points immediately above | Optional deletion W1; retain the preceding no-guarantee statement |
| Conclusion final sentence | “PACE therefore provides evidence for complementary peer and local detection under benign-only target adaptation, with independent operational calibration and deployment validation remaining necessary.” | Stacked nominal phrases | W2 supplies a shorter sentence preserving the contribution and deployment qualification |

These are stylistic/scientific clarity edits, not evidence that the author used AI. No global rewrite, invisible-mark removal from manuscript files or detector-evasion processing was performed. The declaration must accurately reflect author-confirmed assistance and responsibility under the selected journal’s policy (C1). The audit itself received a deterministic Unicode inspection; that says nothing about authorship of either document.

## 14. Figures, Tables, and Formatting

All 28 PDF pages were rendered and inspected as page contact sheets, with the scientific figures/table/equation pages additionally checked at readable resolution. A fresh build in a temporary directory reproduced identical extracted PDF text. The LaTeX log has no overfull/underfull boxes or undefined references/citations; BibTeX’s single empty-pages warning is addressed proportionately in m22. Body type, equations, cross-references and reference rendering are readable. The PDF is not tagged; no accessibility certification is claimed.

| Asset / page | Evidence-based assessment | Surgical action |
|---|---|---|
| Fig.1 architecture, p.9 | Clear two-route/two-local-tier structure; caption explicitly discloses pooled preprocessing | Keep; supply generator (m13). Additional repeated caveats inside boxes are unnecessary |
| Table 1 prior work, p.4 | Readable methodological comparison, explicitly not a leaderboard | Keep; add a thresholding precedent in text (O7), not a crowded compulsory new table row |
| Table 2 data access, p.7 | Makes N versus total target data explicit | Keep |
| Table 3 corpora, p.13 | Evaluation units and dimensions match the cache schemas | Keep; add release facts in availability (m17) |
| Table 4 primary, p.17 | Correct numbers and appropriate nominal-setting label | Add FPR-difference intervals (M19), and short conditional-interval caption (M14) |
| Table 5 FPR, p.18 | Quantiles/exceedance denominators are clear and reproduce | Keep; optional calibration-size diagnostic belongs alongside, not as an explanation proved by the table |
| Fig.2 target gains, p.19 | Negative outcomes remain visible; shapes supplement colors | Optional labels on scenario-12 and largest gains; keep independent panel ranking explicit (M1) |
| Fig.3 FPR ECDF, p.21 | Full tail visible; lower-rate detail compressed | Optional inset (O5), retain exact zeros and tail |
| Fig.4 budget scaling, p.22 | Fixed cohorts are clearly distinguished; bars are computed correctly but condition on singleton TON contribution | Add M14 caveat or switch the high-resource view to descriptive target changes; no arbitrary min-size rule |
| Appendix complete-arm table | Missing although all rules are defined | Add the complete inventory in §9 (M13), including adverse reserve/anchor tradeoffs |

A new decorative figure is unnecessary. One complete comparison table has greater value than adding more conceptual diagrams. Any figure work recommended here is a future presentation edit; no submitted figure was changed by this audit.

## 15. Independent Reviewer Assessments

These are three distinct simulated reviewer perspectives applied to the independently checked evidence, not claims of external human review or three independent experimental replications.

**Reviewer A — Novelty and literature.** The precise benign-only target regime and retained per-peer decision architecture are useful; the paper already acknowledges that the estimators are established. The likely major objection is incremental assembly without an external competitor or sufficient emphasis on the architecture-specific experiment. Bridges/CADES and the recent multi-source literature sharpen the boundary. Request the complete rule comparisons (M13), make peer fusion visible in results (M5), and obtain MEFA before making exhaustive novelty claims. A protocol-compatible MCDE comparison is the highest-value experimental addition, but numerical comparisons to its published tables would be invalid. Provisional assessment: major revision for positioning and comparative completeness; no demonstrated prior method duplicates the full verified architecture.

**Reviewer B — Methodology and statistics.** Equations, code, current implementation identity and primary numbers are strong. The major concerns are limited operating coverage, conditional singleton intervals and design-target reuse; the complete arm inventory is essential. Ask for freeze/arm chronology, explicit endpoint interpretation and high-resource host sensitivity (M12–M16). Do not accept zero-credit clipping as a lower bound or sign-test p-values as a cure for shared-peer dependence. The .0125 allocation is disclosed; its allocation-respecting sensitivity is useful but not an identity proof (m20). Provisional assessment: major revision, mainly manuscript and existing-results work; no demonstrated implementation repair is necessary.

**Reviewer C — Security relevance and practice.** Trusted benign onboarding with labeled peer knowledge is a meaningful operational constraint, and reporting the FPR tail is a strength. Practical evidence remains limited: trusted/model-compatible peers, centralized preprocessing, delayed Gotham aggregates, mostly seen-label transfer and no measured system/incident workload. The .67 FPR is serious even when shared with baselines. Request a precise deployment scope and the existing role-shift/point-mass diagnostics (M17, M19), retaining causal uncertainty. Latency, poisoning and adaptive recalibration are not mandatory defects in a paper that explicitly excludes them; they become necessary if deployment/security claims broaden. Provisional assessment: a bounded offline methods contribution is credible after the reporting revisions; deployment readiness remains unestablished.

**Associate-editor synthesis.** C1 blocks submission administratively. M13, M12, M14 and M16 are credible causes of rejection if reviewers perceive incomplete comparisons or an overbroad inferential claim. M5 and M1 determine whether the legitimate contribution is visible and proportionate. Existing disclosure already resolves objections that the paper promises model-only preprocessing, 100 total onboarding observations, independent physical devices, guaranteed nominal FPR or unseen families everywhere; those are not the paper’s current claims. An external baseline/independent cohort could be requested by a journal, but there is no basis to demand an algorithm redesign simply because FPR is heterogeneous. Prioritize a scientific reporting revision; judge the need for new experiments against the chosen venue and final claim scope.

## 16. Reviewer Objections and Proposed Responses

| Credible objection | Evidence-backed response and limit |
|---|---|
| “Peers alone outperform your comparator.” | Correct at α=.05. Report both simpler arms and show PACE’s joint mean TPR/FPR tradeoff (M13); its detection interval against peers-only includes zero |
| “This is only a larger ensemble.” | Fixed-local-bank fusion arms test peer decision structure. Report nominal gains **with** FPR cost and avoid exact matched-FPR or mechanism claims (M5,M12) |
| “You claim matched FPR from uncovered points.” | The current paper already says approximate clamped adjustment; add direction and nonidentification. Zero-credit clipping is a sensitivity, not a bound (M12) |
| “The small-cohort interval is implausibly narrow.” | Its arithmetic is correct, but the singleton corpus cannot vary; show the dominant host and leave-one-host-out interpretation (M14) |
| “Your method spends .0125 at α=.01.” | Explicitly disclosed. The allocation-respecting single-union alternative preserves pooled detection at stored precision with close FPR; no identical-row-mask claim (m20) |
| “The FPR reduction is only two targets.” | Much of its magnitude at N=400 is two targets, but excluding them leaves −.002107 with a negative conditional interval; preserve and qualify the correct mean claim (M19) |
| “The anchor is just a development repair.” | Its motivating target is documented; report this and the ablation’s simultaneous bank-budget change. It is a design choice with observed tradeoffs, not a worst-case guarantee (M16,M20) |
| “Gains are a TON artefact.” | Ex-TON primary gains remain positive and small; concentration, negative larger-budget findings and outcome-selected sensitivities are all reported (M1) |
| “FPR .67 is operationally unacceptable.” | The benchmark establishes no deployment guarantee. Shared baseline failures and benign-role shifts localize the limitation without excusing it or proving a fix (M17) |
| “Your target has already been used in design.” | Target reuse is disclosed; add verifiable decision/arm chronology. New seeds assess randomness on reused targets, not external generalization (M16) |
| “This is not unseen-family detection.” | Correct: 11/258 label cells are absent from peers, with label identity not equivalent to attack behavior. The target has no attack labels at detector fitting, a distinct claim (m16) |

## 17. Recommended Additional Analyses and Experiments

A = manuscript only; B = analysis of existing artifacts; C = new experiment; D = code/protocol change. None of C or D was executed in this audit. Effort estimates are planning ranges, not measured promises.

| ID / class | Scientific question and exact action | Existing evidence / benefit | Priority, effort, submission role |
|---|---|---|---|
| A1 / A | Publish all recorded rules with TPR/FPR and paired effects | §9 supplies values; resolves selective-reporting concern without assuming intent | P1; 0.5–1 day; essential M13 |
| A2 / A | State endpoint bias/nonidentification and singleton conditioning | M12/M14 calculations available; corrects inference, not printed data | P1; 0.5 day; essential |
| B1 / B | Export target×seed×arm summaries, preserving all negative outcomes | 1,416 unique stored records; improves transparency of regressions and budget sensitivity | P2; hours; recommended release supplement |
| B2 / B | Show operating-coverage counts and a covered-only sensitivity with changed cohort denominators | Stored sweeps permit it. Covered subsets are selected by outcomes and cannot replace the full primary estimator | P2; hours; optional, may weaken gains |
| B3 / B | Show leave-top-five-out and leave-singleton-host-out descriptions with selection labels | Calculations verified; tests concentration without claiming independent validation | P1 for M14, P2 for top-five; hours |
| B4 / B | Plot FPR against exact calibration count and mark corpus/point-mass targets | Cache roles and outcomes available; tests associations, not whether more calibration causes improvement | P2; 0.5 day; optional figure |
| B5 / B | Add equal-corpus macro and per-seed target summaries | Different estimand and shared-peer caveats explicit; may strengthen or challenge pooled interpretation | P3; hours; optional |
| E1 / C | Refit/re-evaluate a wider **benign-selected** operating sweep for all claimed competitors, bracketing comparator FPRs | Four points currently insufficient. State selection and decision randomization rules; target test labels cannot select a new deployed threshold | P1 only if exact/strong matched-FPR superiority is claimed; days–week; otherwise optional |
| E2 / C | Run a documented MCDE reimplementation with identical target roles/source access and unchanged reporting units | Closest hybrid baseline; could strengthen or reverse comparative conclusions; report all outcomes | P1 reviewer risk; roughly 1–2 weeks; venue-dependent, no superiority assumption |
| E3 / C | Reserve unseen clients/strata before tuning; train appropriate compatible source peers without using held-out target outcomes | New seeds on reused targets do not answer this question. Entire-corpus transfer requires a common feature schema or separately defined within-corpus onboarding test | P2; days–weeks plus data access; essential only for external-generalization claims |
| E4 / C | Additional seeds on randomness-sensitive targets and a prespecified aggregation/interval | Quantifies seed variation on .190/scenario-12; cannot remove cohort-selection bias | P2; campaign-dependent; optional |
| E5 / C | Instrument per-peer score/alert traces for scenario 11 and the motivating drowning examples | Current rates identify branches, not causal experts. Requires later scoring instrumentation; no such re-scoring performed here | P2; days; optional mechanism study |
| E6 / C | Measure latency/memory/throughput and incident aggregation on a named deployment system | Algebraic parameter counts are insufficient; result may challenge practical value | P2 if systems claims; device-dependent; otherwise optional |
| D1 / D | Any proposed changes to resolution, pooling, trust or recalibration need a separately specified protocol and fresh evaluation | No correctness defect requires them now; changing a rule to improve these same targets compounds selection | P3 extension; scope-dependent; not required for current bounded claim |

For E1, adjusting to a comparator’s held-out FPR is a retrospective evaluation, not a label-free deployment policy; report the distinction. For E3, “leave one corpus out” cannot mean directly passing incompatible feature vectors through the same MLP. New experiments can improve or weaken the story and must be reported either way.

## 18. Surgical Manuscript Fixes

Apply stable finding IDs as the authoritative edit specification. Source lines refer to this unchanged draft; use the quoted anchors because later edits shift line numbers. These actions are recommendations only.

| Order / source | Exact operation | Finding / verification |
|---|---|---|
| 1. main.tex:17,25,35–42 | Replace author/declaration placeholders with author-approved facts; remove draft banner after approval | C1; no invented release or tool history |
| 2. experimental-methodology.tex:31–35 and results.tex after the primary comparison | Add the complete-arm appendix/table and the peers-only paragraph in M13 | All 12 arms plus two recorded baselines, denominators and FPRs |
| 3. results.tex:64, after the endpoint-count paragraph | Insert the M12 endpoint-direction/nonidentification paragraph | Avoid exact-matching/bound language |
| 4. results.tex:71,77 and Figure 4 caption | Insert the dominant-host and singleton-interval explanation from M14; annotate bars or show descriptive target changes | Preserve the actual high-resource regression |
| 5. experimental-methodology.tex:50 and limitations.tex:4 | Add author-substantiated design/criterion/arm chronology, using M16’s factual boundary | No unsupported exact private-session timing |
| 6. discussion.tex:13 → a §6 results subsection | Move the peer-fusion values with their FPRs; retain one discussion cross-reference and M5’s interpretation | No duplicate long results paragraph |
| 7. introduction.tex:14 and main.tex:19 | Give the peer-wise contrast appropriate scope; insert only M5’s concise abstract sentence if space permits | Primary mean-local comparison remains explicit |
| 8. results.tex:27 | Replace “two lower bounds lie very close to zero” with M1’s numeric concentration wording | All four ex-TON intervals retained |
| 9. results.tex:22 and discussion.tex:8 | Add M20’s bank-budget-reallocation/35-1 detail; retain the correct α=.05 tradeoff | No “worst-case protection” theorem implied |
| 10. Table 4 lower panel; results.tex:6 | Add M19 FPR intervals and two-target/exclusion detail; retain lower-mean-FPR headline/highlight | No compulsory “slightly” substitution |
| 11. results.tex:50 and §6.3 | Add selected M17/M18/m21 diagnostics; retain causal uncertainty and regressions | Associations and headroom are descriptive |
| 12. methodology.tex:81 | Connect single-union sensitivity (m20); optionally give both resolution formulas (m15) | Never infer equal alert masks from equal rates |
| 13. experimental-methodology.tex:19 | Add label-based 11/258 coverage sentence (m16) | Not a family-holdout claim |
| 14. availability and experimental-methodology.tex:4 | Name exact archives/subsets/access conditions; add actual Gotham descriptor and N-BaIoT data DOI | m17; do not paste unverified raw checksums |
| 15. related-work.tex §§2.2,2.4,2.6 | Add only the exact, claim-specific O4/O6/O7/O9 citations selected from §11 | Preserve current correct Xiang description; no fabricated DOI |
| 16. introduction.tex:7; optional abstract/conclusion | Apply W7, then optional W1/W2; M5 contains the fourth wording opportunity | Precise transformation and preserved scientific meaning |
| 17. release figure sources / README / docs | Later authorized figure/documentation bookkeeping (m13,m19,O5), outside frozen campaign code | Not an algorithm correction |

No whole-abstract rewrite is required. In particular, keep the validated primary gains, lower α=.05 mean FPR, α=.01 alert cost, negative subgroups, anchor tradeoff, explicit .0125 allocation exception and external-comparison disclaimer. Accurate formal sentences should not be replaced solely to change style.

## 19. Final Prioritized Implementation Plan

1. **Essential correctness and completeness:** authors complete C1; report all executed rules (M13); add M12’s endpoint interpretation and M14’s singleton explanation. No algorithm changes are required by demonstrated evidence.
2. **Scientific/statistical strengthening:** substantiate chronology (M16); report concentration and bank-budget reallocation (M1,M20); add the already computed FPR, win/loss, label-coverage and seed-localization details selectively (M17–M19,m16,m21). Preserve denominators and causal uncertainty.
3. **Novelty and recent literature:** put the fusion experiment in results (M5), cite ensemble-thresholding and close transfer work (O6,O7), select current evaluation sources (O4,O9), and resolve MEFA access before making an exhaustive novelty statement. Treat current preprint versions accurately.
4. **Results presentation:** complete-arm appendix, FPR intervals, conditional Figure 4 caption, data/input provenance and figure generators (m17,m13). An optional low-FPR inset can follow.
5. **Wording and visual polish:** W7 first; W1/W2 only if useful; apply the chosen journal’s reference/layout style (m22,O2). Align release docs in a separately authorized edit (m19).
6. **Optional experiments/extensions:** choose E1/E2/E3 against the intended claim and venue. Additional seeds, mechanism traces and systems measurements have narrower purposes. Prespecify the new protocol and report unfavorable outcomes. Do not tune this frozen rule merely to obtain a better headline.

## 20. Final Verification and Submission Readiness

| Pass | Completed evidence and limits |
|---|---|
| A — Reconciliation | Every substantive stable finding, wording proposal and recommended-experiment family in the current audit was adjudicated against available primary evidence. Incorrect or redundant proposed edits are excluded rather than appended as another review. Unverifiable chronology/access remains explicit. |
| B — Science/code/data | Equations, decision rules, roles, seeds, standardizer, own-model exclusion, bank/anchor construction, split eligibility and key adapter paths inspected. Synthetic formula checks establish counterexamples to blanket invariance and distinguish count resolutions. No model fitting occurred. Raw duplicate/provenance verification is not exhaustive. |
| C — Experimental accuracy | 110 artifact content digests valid; current implementation digest matches; 1,824 records, 1,416 unique keys, 408 identical repeats; primary/arm intervals independently recomputed; 4,528 target/scaling/seed/FPR/decomposition scalar checks agree at printed precision. Key host, point-mass, seen-label, endpoint and high-resource sensitivities independently checked. No new scientific campaign. |
| D — References | All 29 cited DOI identities verified; 23 cited PDFs valid; Xiang full HTML supports the current description. Relevant passages in all existing local paper PDFs and the new sources inspected. Metadata, PDF version and claim support distinguished. Four cited article PDFs still unavailable. Publisher/correction coverage is incomplete despite 42 successful Crossref update checks; the TESSERACT erratum is recorded. |
| E — Recent literature | Fresh web, scholarly-index and identifier searches cover 2024–20 October 2026. Nouman DOI/PDF verified; MEFA DOI/abstract verified but full text unavailable; current Youssef/Li preprint identifiers and versions checked. No exhaustive absence-of-prior-art claim. |
| F — Editorial consistency | One entry per issue, stable IDs for surviving underlying issues, 20 requested sections, exact useful replacements, correct units and cross-references. No obsolete review labels, duplicate audit, invented AI/plagiarism score or universal DOI claim. |
| G — Reviewer readiness | Three distinct simulated perspectives and evidence-backed responses; submission requirements distinguished from optional experiments. Accurate favorable results retained alongside negative outcomes. |
| H — Integrity | Baseline content hashes and Git status recorded before investigation. Final comparison covers 589 inspected project files outside .git/.venv/raw-data/client-cache/reference collection; only canonical audit content is intentionally replaced. Reference additions are individually validated. No manuscript, code, config, test, result, dataset or Git-history mutation is authorized or performed. |

**Reference collection:** 40 valid PDFs: 23 cited articles, 14 proposed papers, two contextual preprints and one accurately named supporting thesis. Four files were newly downloaded; all 36 existing PDFs were reused. The new files are `Proposed_14_Nouman_2026.pdf`, `Context_Youssef_2026_v3.pdf`, `Context_Li_2026_v1.pdf`, and `Supporting_Flood_2025_Thesis.pdf`. The two contextual preprints generate Poppler embedded-font type warnings but parse and render; no missing or unreadable text was established. No invalid/uncertain existing file was removed.



| Added reference file | Pages | Source | SHA-256 |
|---|---:|---|---|
| `Proposed_14_Nouman_2026.pdf` | 45 | [Primary repository PDF](https://mdpi-res.com/d_attachment/futureinternet/futureinternet-18-00447/article_deploy/futureinternet-18-00447.pdf) | `055b86c529fcf788e9adc495dce71f8208b35170f6aca7aa9d2ba4325e34856e` |
| `Context_Youssef_2026_v3.pdf` | 24 | [Primary repository PDF](https://arxiv.org/pdf/2605.24696v3) | `b8b73a69f79934b73e593c8c24100fd1f1bdb97a8750fda227cdbb5560e9e634` |
| `Context_Li_2026_v1.pdf` | 12 | [Primary repository PDF](https://arxiv.org/pdf/2609.19241v1) | `c05fe6ce9e73c26f59e10b475efaedf31de65dcb0344a79950da0b218117bc81` |
| `Supporting_Flood_2025_Thesis.pdf` | 205 | [Primary repository PDF](https://era.ed.ac.uk/server/api/core/bitstreams/f235af63-4123-4f6a-a795-433e8b930ec7/content) | `cb2de8cbb0bad50063ea8eeb91785bf6aebb3c9827bfb00922255a50df487731` |

**Access limitations:** article PDFs remain unavailable for cited Refs 10,12,23,25, proposed R4,R7,R15 and candidate MEFA (eight article PDFs). Xiang’s complete HTML and Flood’s supporting thesis are readable, so PDF unavailability is not equated with inability to inspect any full text. Complete article-method verification remains limited for LOF, TON_IoT, CTU-13, Vasilomanolakis, Wardana and MEFA (six records); LOF’s narrow attribution has indexed primary support. The official TESSERACT one-page erratum was read online but its local download was blocked; it is not counted as a missing research article or as a stored PDF.

**Remaining material uncertainties:** exact private development/arm timing and author involvement/approval; inaccessible full methods (particularly the close MEFA neighbor); per-row/per-expert causes of false alerts and literal alert-mask equality; raw-capture duplicate/overlap checks and all archive checksums/licences; deployment generalization, operational incident burden and system resources. The absence of Crossref update notices is not a universal retraction clearance. These limitations are recorded, not represented as passed checks.

**Submission decision:** proceed first with the concrete manuscript-only reporting corrections and author confirmations. The current evidence supports a bounded architectural benchmark paper; exact matched-FPR superiority, independently validated new-target generalization and deployment guarantees remain unsupported. No promise of journal acceptance is made.

