#!/usr/bin/env python3
"""
build_typst_manuscript.py
Compiles manuscript/main_paper.md into a professional two-column JAI UMY camera-ready PDF
using Pandoc and Typst.
"""

import os
import re
import subprocess
import sys

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manuscript_dir = os.path.join(repo_root, "manuscript")
    md_file = os.path.join(manuscript_dir, "main_paper.md")
    body_typ = os.path.join(manuscript_dir, "main_paper_body.typ")
    sections_typ = os.path.join(manuscript_dir, "main_paper_sections.typ")
    camera_ready_typ = os.path.join(manuscript_dir, "main_paper_JAI_camera_ready.typ")
    pdf_out = os.path.join(manuscript_dir, "main_paper_JAI.pdf")

    print("[1/4] Running Pandoc markdown -> typst conversion...")
    pandoc_cmd = [
        "pandoc",
        md_file,
        "--resource-path=manuscript:.",
        "-t", "typst",
        "-o", body_typ
    ]
    subprocess.run(pandoc_cmd, cwd=repo_root, check=True)

    print("[2/4] Processing typst AST and extracting sections...")
    with open(body_typ, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Normalize image paths to repo root
    content = content.replace('"../results/', '"/results/')

    # 2. Split between Frontmatter and Section 1
    split_marker = "== 1. INTRODUCTION"
    if split_marker not in content:
        raise ValueError(f"Could not find '{split_marker}' in converted typst body.")

    frontmatter, body = content.split(split_marker, 1)
    body = "== 1. INTRODUCTION" + body

    # 3. Clean up figure and table captions so Typst does not duplicate "Figure X: Figure X." or "Table X: Table X."
    body = re.sub(r'caption:\s*\[\s*(?:Figure|Table)\s+[\w\d\.\s]+?[\.:]\s*', 'caption: [', body)

    # 4. Optimize table column widths for publication readability
    # Table 1: Model Architecture (20%), DA (8%), Macro F1 (8%), ROC-AUC (8%), MCC (7.5%), Brier (7.5%), Net Return (9%), Sharpe (8%), MDD (10%), Trades (7%), Friction Fee (8.5%)
    t1_old = "columns: (7.41%, 9.26%, 9.26%, 9.26%, 9.26%, 9.26%, 9.26%, 9.26%, 9.26%, 9.26%, 9.26%),"
    t1_new = "columns: (20%, 8%, 8%, 8%, 7.5%, 7.5%, 9%, 8%, 10%, 7%, 8.5%),"
    body = body.replace(t1_old, t1_new)

    # Table 3: Threshold validation optimization (Operational Selection Status given 19.5% width)
    t3_old = "columns: (9.26%, 9.26%, 9.26%, 9.26%, 9.26%, 9.26%, 9.26%, 9.26%, 9.26%, 9.26%, 7.41%),"
    t3_new = "columns: (9%, 10%, 7.5%, 8.5%, 8.5%, 6.5%, 7.5%, 8%, 8.5%, 6.5%, 19.5%),"
    body = body.replace(t3_old, t3_new)

    # Table 2: Ablation study (Model col 26%, Partition 16%, metrics 10% each)
    t2_pattern = r'(columns:\s*\(10\.26%,\s*12\.82%,\s*12\.82%,\s*12\.82%,\s*12\.82%,\s*12\.82%,\s*12\.82%,\s*12\.82%\),)(\s*align:[^;]+?\[Selection\s+Partition\],\s*\[Net Return \(%\)\])'
    body = re.sub(t2_pattern, r'columns: (26%, 16%, 10%, 10%, 10%, 8%, 10%, 10%),\2', body)

    # Table 4: Out-of-sample test evaluation (Model col 25%, Partition 16%)
    t4_pattern = r'(columns:\s*\(10\.26%,\s*12\.82%,\s*12\.82%,\s*12\.82%,\s*12\.82%,\s*12\.82%,\s*12\.82%,\s*12\.82%\),)(\s*align:[^;]+?\[Selection\s+Partition\],\s*\[Net Cum\. Ret \(%\)\])'
    body = re.sub(t4_pattern, r'columns: (25%, 16%, 11%, 8.5%, 10%, 7.5%, 11%, 11%),\2', body)

    # 5. Enhance References section with hanging indent and attach label cleanly
    body = re.sub(
        r'== 6\. REFERENCES\s*\n\s*<references>',
        '== 6. REFERENCES <references>\n#set par(first-line-indent: 0pt, hanging-indent: 0.5cm)',
        body
    )

    # Write out sections_typ
    with open(sections_typ, "w", encoding="utf-8") as f:
        f.write("// Auto-generated body sections from main_paper.md\n")
        f.write("#let divider() = v(2pt)\n\n")
        f.write(body)

    print("[3/4] Creating camera-ready Typst master document...")
    camera_ready_content = """// JAI UMY Camera-Ready Master Document
#import "jai_template.typ": *

#let divider() = v(2pt)

#show: jai_article.with(
  title: [Downside Risk Mitigation in Cryptocurrency Markets: A Dual-Stream Gated Hybrid LSTM Framework under Market Frictions],
  authors: [[Author Names Redacted for Double-Blind Peer Review]],
  target_journal: [Journal of Accounting and Investment (JAI - SINTA 2 Accredited)],
  classification: [Empirical Research Article (Computational Finance & Asset Pricing)],
  abstract: [
    *Objective:* This study investigates downside tail-risk mitigation and transaction friction erosion in algorithmic cryptocurrency portfolio management under market microstructure frictions. High-frequency models often succumb to the "Turnover Trap," where excessive trading commissions erode net predictive returns. \\
    *Method:* We propose the _ETH-Sentinel Hybrid LSTM_, a dual-stream architecture decoupling high-frequency price dynamics from behavioral sentiment (Crypto Fear & Greed Index) via an adaptive Gated Cross-Modality Fusion Unit (GFU). To eliminate test-set snooping, a confidence dead-band hysteresis policy ($delta in [0.00, 0.10]$) is calibrated exclusively on validation data ($delta^*_("val") = 0.01$) before single-pass evaluation on out-of-sample Ethereum test bars (fee: 10 bps). \\
    *Results:* In an unhedged execution regime ($delta = 0.00$), sentiment fusion functions as an asymmetric crash cushion, compressing Maximum Drawdown from -24.42% (Technical LSTM) to -18.82% (a 5.60 percentage points tail-risk reduction). Under active validation-selected hysteresis ($delta^* = 0.01$), decision inertia shifts sentiment's marginal role toward turnover suppression: slashing trading frequency by 60.0% (40 to 16 trades), preserving net alpha with a Compounded Return of +15.99%, an Annualized Sharpe Ratio of 1.42, and an MDD of -18.19%, outperforming the technical baseline (+15.18%, Sharpe 1.29). \\
    *Implications/Originality:* This study provides a verifiable selective-abstention framework for digital asset corporate treasury management, demonstrating that behavioral sentiment mitigates catastrophic crashes under unhedged regimes and curbs commission drag under hysteresis execution.
  ],
  keywords: [Cryptocurrency; Dual-Stream LSTM; Sentiment Analysis; Downside Risk; Hysteresis Policy; Market Frictions.],
  jel: [C45; C58; G11; G14.],
)

#include "main_paper_sections.typ"
"""
    with open(camera_ready_typ, "w", encoding="utf-8") as f:
        f.write(camera_ready_content)

    print("[4/4] Compiling camera-ready PDF via Typst...")
    typst_cmd = [
        "typst",
        "compile",
        "--root", repo_root,
        camera_ready_typ,
        pdf_out
    ]
    res = subprocess.run(typst_cmd, cwd=repo_root, capture_output=True, text=True)
    if res.returncode != 0:
        print("Typst stderr:\n", res.stderr)
        sys.exit(1)

    print(f"[SUCCESS] Camera-ready PDF compiled successfully: {pdf_out}")

if __name__ == "__main__":
    main()
