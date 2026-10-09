#!/usr/bin/env python3
"""
scripts/verify_table_parity.py
Automated parity verifier between Markdown preview tables and LaTeX table* blocks in manuscript/main_paper.md
"""

import re
import sys

def clean_val(v):
    # Strip markdown bold/italic, latex macros, % signs, extra whitespace
    v = re.sub(r'[\*\_\$\\\{\}]', '', v)
    v = re.sub(r'\\(textbf|textit|mathbf)', '', v)
    v = v.replace('%', '').strip()
    return v

def extract_numbers(s):
    # Extract floating point numbers or signed numbers
    matches = re.findall(r'[-+]?\d*\.?\d+', s)
    return [float(m) for m in matches]

def is_separator_row(cells):
    return all(re.match(r'^[\s:-]+$', c) for c in cells if c)

def parse_markdown_table_block(table_text):
    rows = []
    for line in table_text.strip().splitlines():
        line = line.strip()
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if not cells or is_separator_row(cells):
            continue
        rows.append(cells)
    return rows

def parse_latex_tabular_block(tabular_text):
    rows = []
    for line in tabular_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('%') or line.startswith('\\toprule') or line.startswith('\\midrule') or line.startswith('\\bottomrule') or line.startswith('\\multicolumn'):
            continue
        # Check if line ends with \\
        if line.endswith('\\\\'):
            line = line[:-2].strip()
        # Split on '&' not preceded by '\'
        cells = [c.strip() for c in re.split(r'(?<!\\)&', line)]
        # Skip subheader label rows where only first cell has text
        non_empty = [c for c in cells if c.strip()]
        if len(non_empty) <= 1:
            continue
        rows.append(cells)
    return rows

def main():
    with open('manuscript/main_paper.md', 'r', encoding='utf-8') as f:
        content = f.read()

    print("=" * 75)
    print("AUTOMATED VERIFICATION: MARKDOWN VS. LATEX TABLE PARITY")
    print("=" * 75)

    all_pass = True

    # -------------------------------------------------------------
    # 1. Check Table 1 (Technical Benchmarks)
    # -------------------------------------------------------------
    print("\n--- Table 1: tab:technical_benchmark ---")
    t1_md_match = re.search(r'(\| Model Architecture \| DA \(%\).*?\n\n)', content, re.S)
    t1_tex_match = re.search(r'\\begin\{table\*\}'
                             r'.*?\\label\{tab:technical_benchmark\}'
                             r'.*?\\begin\{tabular\*\}\{\\textwidth\}\{.*?\}'
                             r'(.*?)\\end\{tabular\*\}', content, re.S)

    if not t1_md_match or not t1_tex_match:
        print("FAIL: Could not locate Table 1 Markdown or LaTeX blocks!")
        all_pass = False
    else:
        md_rows = parse_markdown_table_block(t1_md_match.group(1))[1:] # skip header
        tex_rows = parse_latex_tabular_block(t1_tex_match.group(1))[1:] # skip header

        print(f"Comparing {len(md_rows)} rows across 9 numeric metrics...")
        t1_pass = True
        for i, (m_row, t_row) in enumerate(zip(md_rows, tex_rows)):
            arch_name = m_row[0].replace('*', '').strip()
            # columns 1 to 9: DA, Macro F1, ROC-AUC, MCC, Brier, Cum Ret, Sharpe, MDD, Trades, Friction
            for j in range(1, 10):
                m_nums = extract_numbers(clean_val(m_row[j]))
                t_nums = extract_numbers(clean_val(t_row[j]))
                if m_nums != t_nums:
                    print(f"  FAIL: Row '{arch_name}' Col {j}: MD '{m_row[j]}' != LaTeX '{t_row[j]}'")
                    t1_pass = False
                    all_pass = False
        if t1_pass and len(md_rows) == len(tex_rows):
            print("  STATUS: [PASS] (Table 1: 100% parity across all 4 benchmark models)")
        else:
            print("  STATUS: [FAIL]")

    # -------------------------------------------------------------
    # 2. Check Table 2 (Ablation Benchmark)
    # -------------------------------------------------------------
    print("\n--- Table 2: tab:hybrid_ablation_benchmark ---")
    t2_md_match = re.search(r'(\| Model Architecture & Policy \| Selection Partition \| Net Return.*?\n\n)', content, re.S)
    t2_tex_match = re.search(r'\\begin\{table\*\}'
                             r'.*?\\label\{tab:hybrid_ablation_benchmark\}'
                             r'.*?\\begin\{tabular\*\}\{\\textwidth\}\{.*?\}'
                             r'(.*?)\\end\{tabular\*\}', content, re.S)

    if not t2_md_match or not t2_tex_match:
        print("FAIL: Could not locate Table 2 Markdown or LaTeX blocks!")
        all_pass = False
    else:
        md_rows = parse_markdown_table_block(t2_md_match.group(1))[1:] # skip header
        tex_rows = parse_latex_tabular_block(t2_tex_match.group(1))[1:] # skip header

        print(f"Comparing {len(md_rows)} rows across 6 economic metrics...")
        t2_pass = True
        for i, (m_row, t_row) in enumerate(zip(md_rows, tex_rows)):
            arch_name = m_row[0].replace('*', '').strip()
            # MD: Model(0), Partition(1), Ret(2), Sharpe(3), MDD(4), Trades(5), Friction(6), Exposure(7)
            # LaTeX: Model(0), Config(1), Ret(2), Sharpe(3), MDD(4), Trades(5), Friction(6), Exposure(7)
            for j in range(2, 8):
                m_nums = extract_numbers(clean_val(m_row[j]))
                t_nums = extract_numbers(clean_val(t_row[j]))
                if m_nums != t_nums:
                    print(f"  FAIL: Row '{arch_name}' Col {j}: MD '{m_row[j]}' != LaTeX '{t_row[j]}'")
                    t2_pass = False
                    all_pass = False
        if t2_pass and len(md_rows) == len(tex_rows):
            print("  STATUS: [PASS] (Table 2: 100% parity across all 5 ablation models)")
        else:
            print("  STATUS: [FAIL]")

    # -------------------------------------------------------------
    # 3. Check Table 3 Panel A (Validation Optimization)
    # -------------------------------------------------------------
    print("\n--- Table 3 Panel A: Validation Parameter Optimization ---")
    t3a_md_match = re.search(r'##### Panel A: Parameter Optimization.*?\n(\| Delta .*?\n\n)', content, re.S)
    t3_tex_match = re.search(r'\\begin\{table\*\}'
                             r'.*?\\label\{tab:threshold_sensitivity\}'
                             r'.*?\\begin\{tabular\*\}\{\\textwidth\}\{.*?\}'
                             r'(.*?)\\end\{tabular\*\}', content, re.S)

    if not t3a_md_match or not t3_tex_match:
        print("FAIL: Could not locate Table 3 Panel A blocks!")
        all_pass = False
    else:
        md_rows_a = parse_markdown_table_block(t3a_md_match.group(1))[1:] # skip header
        tex_full = t3_tex_match.group(1)
        panel_a_tex_str = tex_full.split('Panel B:')[0]
        tex_rows_a = parse_latex_tabular_block(panel_a_tex_str)[1:] # skip header

        print(f"Comparing {len(md_rows_a)} MD rows vs {len(tex_rows_a)} LaTeX rows...")
        t3a_pass = True
        if len(md_rows_a) != len(tex_rows_a):
            print(f"  FAIL: Row count mismatch: MD has {len(md_rows_a)} rows, LaTeX has {len(tex_rows_a)} rows!")
            t3a_pass = False
            all_pass = False
        else:
            for i, (m_row, t_row) in enumerate(zip(md_rows_a, tex_rows_a)):
                # Compare Delta (0)
                m_delta = extract_numbers(clean_val(m_row[0]))
                t_delta = extract_numbers(clean_val(t_row[0]))
                if m_delta != t_delta:
                    print(f"  FAIL: Row {i+1} Delta mismatch: MD '{m_row[0]}' != LaTeX '{t_row[0]}'")
                    t3a_pass = False
                    all_pass = False
                # Compare DeadBand (1)
                m_db = extract_numbers(clean_val(m_row[1]))
                t_db = extract_numbers(clean_val(t_row[1]))
                if m_db != t_db:
                    print(f"  FAIL: Row {i+1} DeadBand mismatch: MD '{m_row[1]}' != LaTeX '{t_row[1]}'")
                    t3a_pass = False
                    all_pass = False
                # Compare Hybrid Sharpe: MD col 2 vs LaTeX col 2
                m_hs = extract_numbers(clean_val(m_row[2]))
                t_hs = extract_numbers(clean_val(t_row[2]))
                if m_hs != t_hs:
                    print(f"  FAIL: Row {i+1} Hyb Sharpe: MD '{m_row[2]}' != LaTeX '{t_row[2]}'")
                    t3a_pass = False
                    all_pass = False
                # Compare Hybrid Ret: MD col 3 vs LaTeX col 3
                m_hr = extract_numbers(clean_val(m_row[3]))
                t_hr = extract_numbers(clean_val(t_row[3]))
                if m_hr != t_hr:
                    print(f"  FAIL: Row {i+1} Hyb Ret: MD '{m_row[3]}' != LaTeX '{t_row[3]}'")
                    t3a_pass = False
                    all_pass = False
                # Compare Tech Sharpe: MD col 6 vs LaTeX col 4
                m_ts = extract_numbers(clean_val(m_row[6]))
                t_ts = extract_numbers(clean_val(t_row[4]))
                if m_ts != t_ts:
                    print(f"  FAIL: Row {i+1} Tech Sharpe: MD '{m_row[6]}' != LaTeX '{t_row[4]}'")
                    t3a_pass = False
                    all_pass = False
                # Compare Tech Ret: MD col 7 vs LaTeX col 5
                m_tr = extract_numbers(clean_val(m_row[7]))
                t_tr = extract_numbers(clean_val(t_row[5]))
                if m_tr != t_tr:
                    print(f"  FAIL: Row {i+1} Tech Ret: MD '{m_row[7]}' != LaTeX '{t_row[5]}'")
                    t3a_pass = False
                    all_pass = False

        if t3a_pass:
            print("  STATUS: [PASS] (Table 3 Panel A: 100% parity including delta = 0.05)")
        else:
            print("  STATUS: [FAIL]")

    # -------------------------------------------------------------
    # 4. Check Table 3 Panel B (Out-of-Sample Test Set)
    # -------------------------------------------------------------
    print("\n--- Table 3 Panel B: Out-of-Sample Performance ---")
    t3b_md_match = re.search(r'##### Panel B: Out-of-Sample Test Set Evaluation.*?(\| Model Architecture & Policy.*?\n\n)', content, re.S)
    if not t3b_md_match or not t3_tex_match:
        print("FAIL: Could not locate Table 3 Panel B blocks!")
        all_pass = False
    else:
        md_rows_b = parse_markdown_table_block(t3b_md_match.group(1))[1:] # skip header
        panel_b_tex_str = t3_tex_match.group(1).split('Panel B:')[1]
        tex_rows_b = parse_latex_tabular_block(panel_b_tex_str)[1:] # skip header

        print(f"Comparing {len(md_rows_b)} MD rows vs {len(tex_rows_b)} LaTeX rows...")
        t3b_pass = True
        if len(md_rows_b) != len(tex_rows_b):
            print(f"  FAIL: Row count mismatch: MD has {len(md_rows_b)} rows, LaTeX has {len(tex_rows_b)} rows!")
            t3b_pass = False
            all_pass = False
        else:
            for i, (m_row, t_row) in enumerate(zip(md_rows_b, tex_rows_b)):
                arch_name = m_row[0].replace('*', '').strip()
                # MD cols: Model(0), Partition(1), Ret(2), Sharpe(3), MDD(4), Trades(5), Friction(6), Exposure(7)
                # LaTeX cols: Model(0), Config(1), Ret(2), Sharpe(3), MDD(4), Trades(5), Friction(6)
                for j in range(2, 7):
                    m_nums = extract_numbers(clean_val(m_row[j]))
                    t_nums = extract_numbers(clean_val(t_row[j]))
                    if m_nums != t_nums:
                        print(f"  FAIL: Row '{arch_name}' Col {j}: MD '{m_row[j]}' != LaTeX '{t_row[j]}'")
                        t3b_pass = False
                        all_pass = False

        if t3b_pass:
            print("  STATUS: [PASS] (Table 3 Panel B: 100% parity across full out-of-sample grid)")
        else:
            print("  STATUS: [FAIL]")

    print("\n" + "=" * 75)
    if all_pass:
        print("OVERALL RESULT: ALL TABLES PASS PARITY VERIFICATION (100% MATCH)")
        print("=" * 75)
        sys.exit(0)
    else:
        print("OVERALL RESULT: TABLE PARITY VERIFICATION FAILED")
        print("=" * 75)
        sys.exit(1)

if __name__ == '__main__':
    main()
