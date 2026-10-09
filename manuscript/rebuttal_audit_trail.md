# Internal Peer-Review Audit Trail (Rounds 1–4)

This document summarizes the chronological resolution of methodological issues raised during four internal review rounds, provided for editorial and reviewer transparency.

## Round 1 — Initial Methodological Audit
Issues identified: test-set snooping in hysteresis threshold ($\delta^*$) selection; absence of formal statistical significance testing; lack of a symmetric ablation baseline for the Technical LSTM; an arithmetic inconsistency in the reported bar-count reconciliation (4,370 vs. 4,357); inconsistent EMA half-life derivation; an unverifiable citation.

## Round 2 — First Remediation
Resolutions: validation-only calibration of $\delta^*$ introduced; symmetric ablation added for the Technical LSTM baseline; stationary block bootstrap significance testing introduced; bar-count reconciled (13 warm-up bars removed, 4,357 clean bars); EMA half-life corrected to ≈2.06 bars (8.24 hours).  
Issues remaining: selection logic for $\delta^*=0.01$ not yet formalized against an explicit criterion; MDD reversal at $\delta^*=0.01$ (Hybrid worse than Technical) not yet discussed; residual overclaiming despite non-significant bootstrap p-value; unit mismatch in the treasury cost case study; citation year/DOI inconsistency.

## Round 3 — Second Remediation
Resolutions: formal ex-ante constrained optimization objective for $\delta^*$ selection introduced and verified against the validation grid; Hypothesis 1 reformulated as conditional on execution regime, with the MDD reversal openly discussed; statistical language calibrated to explicitly report non-significance ($p=0.9170$, 95% CI crossing zero); treasury cost case study corrected with explicit period vs. annualized distinction and an AMH regime-shift disclaimer; Hendrickx et al. citation corrected and independently verified (Machine Learning, 113(5), 3073–3110, 2024, Springer Nature).  
Issues remaining (found via full cross-document audit): a narrative/table contradiction regarding the $\delta \ge 0.05$ vs. $\ge 0.06$ zero-trade boundary; selective presentation of only three grid points in Table 3 Panel B despite full-grid data existing in the Appendix.

## Round 4 — Final Remediation
Resolutions: $\delta \ge 0.05$/$\ge 0.06$ narrative-table contradiction corrected and cross-partition probability bounds clarified; Table 3 Panel B expanded to disclose the full out-of-sample grid ($\delta \in [0.00, 0.05]$); explicit statistical non-significance qualification added to the closing paragraph of the Hypothesis 2 discussion.

## Final Disposition
ACCEPT, subject to a single typesetting-stage correction (LaTeX/Markdown table row parity for $\delta=0.05$ in Table 3 Panel A), executed as part of this production run.
