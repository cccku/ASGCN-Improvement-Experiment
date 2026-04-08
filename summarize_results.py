#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_results.py
---------------------
Parses all per-seed log files under results/ and prints:
  1. Per-seed results table
  2. Mean ± std summary table (for reporting)
  3. Delta vs paper baseline

Usage:
    python summarize_results.py
    python summarize_results.py --csv
    python summarize_results.py --results_dir path
"""

import os, re, argparse, statistics

PAPER = {
    'rest14': {'acc': 80.86, 'f1': 72.19},
    'lap14':  {'acc': 75.55, 'f1': 71.05},
    'rest15': {'acc': 79.89, 'f1': 61.89},
    'rest16': {'acc': 88.99, 'f1': 67.48},
}

IMPROVEMENTS = [
    ('A_baseline',    'A-Baseline'),
    ('B_bert',        'B-BERT'),
    ('C_gat',         'C-GAT'),
    ('D_softmask',    'D-SoftMask'),
    ('E_contrastive', 'E-Contrastive'),
]

DATASETS = ['rest14', 'lap14', 'rest15', 'rest16']
SEEDS    = [42, 123, 776]


def parse_log(path):
    if not os.path.exists(path):
        return None
    content = open(path, encoding='utf-8', errors='ignore').read()
    seed_m = re.search(r'Seed\s*:\s*(\d+)', content)
    seed   = int(seed_m.group(1)) if seed_m else '?'
    if 'RESULT: FAILED' in content:
        return (seed, 'FAIL', 'FAIL')
    m = re.search(r'RESULT:\s*seed=\d+\s+acc=([0-9.]+)\s+f1=([0-9.]+)', content)
    if m:
        return (seed, float(m.group(1))*100, float(m.group(2))*100)
    accs = re.findall(r'max_test_acc_avg[:\s]+([0-9.]+)', content)
    f1s  = re.findall(r'max_test_f1_avg[:\s]+([0-9.]+)',  content)
    if accs and f1s:
        return (seed, float(accs[-1])*100, float(f1s[-1])*100)
    return None


def mean_std(vals):
    vals = [v for v in vals if v is not None and v != 'FAIL']
    if not vals: return None, None
    m = sum(vals)/len(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--csv', action='store_true')
    args = parser.parse_args()
    rdir = args.results_dir

    # Collect data[imp_dir][ds][seed] = (acc, f1)
    data = {}
    for imp_dir, _ in IMPROVEMENTS:
        data[imp_dir] = {}
        for ds in DATASETS:
            data[imp_dir][ds] = {}
            for seed in SEEDS:
                log = os.path.join(rdir, imp_dir, f'{ds}_seed{seed}.txt')
                res = parse_log(log)
                if res:
                    data[imp_dir][ds][seed] = (res[1], res[2])

    SEP = '  ' + '-'*86

    # ── TABLE 1: Per-seed ──────────────────────────────────────────────────
    print('\n' + '='*90)
    print('  TABLE 1: Per-seed Results')
    print('='*90)
    for ds in DATASETS:
        p = PAPER.get(ds, {})
        print(f'\n  Dataset: {ds}   Paper baseline → acc={p.get("acc")}  f1={p.get("f1")}')
        print(f'  {"Improvement":<18}' + ''.join(f'  seed={s} acc    f1   ' for s in SEEDS))
        print(SEP)
        for imp_dir, label in IMPROVEMENTS:
            row = f'  {label:<18}'
            for seed in SEEDS:
                r = data[imp_dir][ds].get(seed)
                if r is None:            row += f'  {"pending":<20}'
                elif r[0] == 'FAIL':     row += f'  {"FAILED":<20}'
                else:                    row += f'  {r[0]:6.2f}  {r[1]:6.2f}      '
            print(row)

    # ── TABLE 2: Mean ± Std ────────────────────────────────────────────────
    print('\n' + '='*90)
    print('  TABLE 2: Mean ± Std across seeds (42, 123, 776)  ')
    print('='*90)
    print(f'\n  {"Improvement":<18}' + ''.join(f'  {ds:^24}' for ds in DATASETS))
    print(f'  {"":18}' + ''.join(f'  {"acc":^11} {"f1":^11} ' for _ in DATASETS))
    print(SEP)

    # Paper row
    print(f'  {"Paper (ASGCN)":<18}' +
          ''.join(f'  {str(PAPER[ds]["acc"]):^11} {str(PAPER[ds]["f1"]):^11} ' for ds in DATASETS))
    print(SEP)

    csv_rows = [['Improvement'] + [f'{ds}_{m}' for ds in DATASETS for m in
                 ['acc_mean','acc_std','f1_mean','f1_std']]]

    for imp_dir, label in IMPROVEMENTS:
        row = f'  {label:<18}'
        csv_row = [label]
        for ds in DATASETS:
            accs = [data[imp_dir][ds][s][0] for s in SEEDS if s in data[imp_dir][ds]]
            f1s  = [data[imp_dir][ds][s][1] for s in SEEDS if s in data[imp_dir][ds]]
            am, as_ = mean_std(accs)
            fm, fs  = mean_std(f1s)
            if am is None:
                row += f'  {"pending":^11} {"pending":^11} '
                csv_row += ['','','','']
            else:
                row += f'  {am:5.2f}±{as_:.2f}  {fm:5.2f}±{fs:.2f} '
                csv_row += [f'{am:.4f}', f'{as_:.4f}', f'{fm:.4f}', f'{fs:.4f}']
        print(row)
        csv_rows.append(csv_row)

    # ── TABLE 3: Delta vs paper ────────────────────────────────────────────
    print('\n' + '='*90)
    print('  TABLE 3: Δ vs Paper Baseline  (tested mean − paper)')
    print('='*90)
    print(f'\n  {"Improvement":<18}' + ''.join(f'  {ds:^24}' for ds in DATASETS))
    print(f'  {"":18}' + ''.join(f'  {"Δacc":^11} {"Δf1":^11} ' for _ in DATASETS))
    print(SEP)

    for imp_dir, label in IMPROVEMENTS:
        row = f'  {label:<18}'
        for ds in DATASETS:
            accs = [data[imp_dir][ds][s][0] for s in SEEDS if s in data[imp_dir][ds]]
            f1s  = [data[imp_dir][ds][s][1] for s in SEEDS if s in data[imp_dir][ds]]
            am, _ = mean_std(accs)
            fm, _ = mean_std(f1s)
            da = f'{am - PAPER[ds]["acc"]:+.2f}' if am else '  --  '
            df = f'{fm - PAPER[ds]["f1"]:+.2f}'  if fm else '  --  '
            row += f'  {da:^11} {df:^11} '
        print(row)

    print('='*90)

    if args.csv:
        import csv
        out = os.path.join(rdir, 'results_summary.csv')
        with open(out, 'w', newline='') as f:
            csv.writer(f).writerows(csv_rows)
        print(f'\n  CSV saved → {out}')

    print()

if __name__ == '__main__':
    main()