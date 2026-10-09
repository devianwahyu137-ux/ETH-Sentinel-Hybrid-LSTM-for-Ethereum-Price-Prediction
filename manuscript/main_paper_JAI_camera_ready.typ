// JAI UMY Camera-Ready Master Document
#import "jai_template.typ": *

#let divider() = v(2pt)

#show: jai_article.with(
  title: [Downside Risk Mitigation in Cryptocurrency Markets: A Dual-Stream Gated Hybrid LSTM Framework under Market Frictions],
  authors: [[Author Names Redacted for Double-Blind Peer Review]],
  target_journal: [Journal of Accounting and Investment (JAI - SINTA 2 Accredited)],
  classification: [Empirical Research Article (Computational Finance & Asset Pricing)],
  abstract: [
    *Objective:* This study investigates downside tail-risk mitigation and transaction friction erosion in algorithmic cryptocurrency portfolio management under market microstructure frictions. High-frequency models often succumb to the "Turnover Trap," where excessive trading commissions erode net predictive returns. \
    *Method:* We propose the _ETH-Sentinel Hybrid LSTM_, a dual-stream architecture decoupling high-frequency price dynamics from behavioral sentiment (Crypto Fear & Greed Index) via an adaptive Gated Cross-Modality Fusion Unit (GFU). To eliminate test-set snooping, a confidence dead-band hysteresis policy ($delta in [0.00, 0.10]$) is calibrated exclusively on validation data ($delta^*_("val") = 0.01$) before single-pass evaluation on out-of-sample Ethereum test bars (fee: 10 bps). \
    *Results:* In an unhedged execution regime ($delta = 0.00$), sentiment fusion functions as an asymmetric crash cushion, compressing Maximum Drawdown from -24.42% (Technical LSTM) to -18.82% (a 5.60 percentage points tail-risk reduction). Under active validation-selected hysteresis ($delta^* = 0.01$), decision inertia shifts sentiment's marginal role toward turnover suppression: slashing trading frequency by 60.0% (40 to 16 trades), preserving net alpha with a Compounded Return of +15.99%, an Annualized Sharpe Ratio of 1.42, and an MDD of -18.19%, outperforming the technical baseline (+15.18%, Sharpe 1.29). \
    *Implications/Originality:* This study provides a verifiable selective-abstention framework for digital asset corporate treasury management, demonstrating that behavioral sentiment mitigates catastrophic crashes under unhedged regimes and curbs commission drag under hysteresis execution.
  ],
  keywords: [Cryptocurrency; Dual-Stream LSTM; Sentiment Analysis; Downside Risk; Hysteresis Policy; Market Frictions.],
  jel: [C45; C58; G11; G14.],
)

#include "main_paper_sections.typ"
