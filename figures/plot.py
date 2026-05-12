"""
Regenerate paper figures (SBC format). Run from the repo root:
    python figures/plot.py
"""
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import ttest_rel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS   = os.path.join(REPO_ROOT, 'results')
FIGS      = os.path.join(REPO_ROOT, 'figures')

SBC = {
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 300,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.5,
}
plt.rcParams.update(SBC)

COL_W = 3.35

PALETTE = {
    'CNN':         '#2166ac',
    'Inception':   '#d6604d',
    'Transformer': '#7b2d8b',
    'Mamba':       '#e08214',
}

ALL_W = [128, 256, 512, 1024]
MS    = {128: '5.8', 256: '11.6', 512: '23.2', 1024: '46.4'}


def _load(name):
    path = os.path.join(RESULTS, name)
    return json.load(open(path)) if os.path.exists(path) else {}


def _save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGS, f'{name}.{ext}'), bbox_inches='tight', dpi=300)
    print(f'Saved {name}.pdf/.png')


def fig1_frame_length_arch(r_cnn_mb, r_mb, r_tr_mb, r_mamba_mb):
    POS     = {w: i for i, w in enumerate(ALL_W)}
    POS_LBL = [f"w{w}\n({MS[w]} ms)" for w in ALL_W]

    def _pts(rfile):
        xs, ys = [], []
        for w in ALL_W:
            v = rfile.get(f"all_w{w}_norm_es", {}).get("mAP", float("nan"))
            if not np.isnan(v):
                xs.append(POS[w]); ys.append(v)
        return xs, ys

    fig, ax = plt.subplots(figsize=(2.8, 2.2))
    for label, rfile, style in [
        ("CNN",         r_cnn_mb,   "o--"),
        ("Inception",   r_mb,       "s-"),
        ("Transformer", r_tr_mb,    "^-"),
        ("Mamba",       r_mamba_mb, "D-"),
    ]:
        xs, ys = _pts(rfile)
        if ys:
            ax.plot(xs, ys, style, color=PALETTE[label],
                    label=f"{label} ({max(ys):.3f})", markersize=5)

    ax.set_xticks(range(len(ALL_W))); ax.set_xticklabels(POS_LBL)
    ax.set_xlabel("Frame length"); ax.set_ylabel("mAP")
    ax.legend(loc="lower right", frameon=False, fontsize=7, handlelength=1.5)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    _save(fig, 'fig1_frame_length_arch')
    plt.show()


def fig2_cv_comparison():
    inc_cv  = _load('inception_multibranch_cv5_results.json')
    tr_cv   = _load('transformer_multibranch_cv5_results.json')
    WINDOWS = [128, 256]

    inc_folds = {w: [inc_cv[f'all_w{w}_norm_es_fold{i}']['mAP'] for i in range(5)] for w in WINDOWS}
    tr_folds  = {w: [tr_cv[f'all_w{w}_norm_es_fold{i}']['mAP']  for i in range(5)] for w in WINDOWS}
    tests     = {w: ttest_rel(inc_folds[w], tr_folds[w]) for w in WINDOWS}

    def _sig(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'n.s.'

    def _bracket(ax, x0, x1, y, label, dy=0.003):
        ax.plot([x0, x0, x1, x1], [y, y+dy, y+dy, y], color='#333333', lw=0.8)
        ax.text((x0+x1)/2, y+dy+0.0008, label, ha='center', va='bottom', fontsize=7.5)

    fig, ax = plt.subplots(figsize=(COL_W, 3.2))
    bw = 0.32
    x  = np.arange(len(WINDOWS))

    for i, (arch, folds_dict, color) in enumerate([
        ('MB-Inception',   inc_folds, PALETTE['Inception']),
        ('MB-Transformer', tr_folds,  PALETTE['Transformer']),
    ]):
        offset = (i - 0.5) * bw
        means  = [np.mean(folds_dict[w]) for w in WINDOWS]
        stds   = [np.std(folds_dict[w])  for w in WINDOWS]
        ax.bar(x + offset, means, bw - 0.04, yerr=stds, label=arch, color=color,
               error_kw=dict(elinewidth=1.2, capsize=3, ecolor='#333333'), zorder=3)

    for i, w in enumerate(WINDOWS):
        t, p  = tests[w]
        inc_m = np.mean(inc_folds[w]); inc_s = np.std(inc_folds[w])
        tr_m  = np.mean(tr_folds[w]);  tr_s  = np.std(tr_folds[w])
        _bracket(ax, i - bw/2, i + bw/2, max(inc_m + inc_s, tr_m + tr_s) + 0.004,
                 f'{_sig(p)}  p={p:.3f}')

    ax.set_xticks(x)
    ax.set_xticklabels([f'w{w}' for w in WINDOWS], fontsize=9)
    ax.set_xlabel('Frame length', fontsize=9)
    ax.set_ylabel('mAP (mean ± std, 5-fold CV)', fontsize=9)
    ax.legend(fontsize=5, framealpha=0.7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    fig.tight_layout()
    _save(fig, 'fig2_cv_comparison')

    for w in WINDOWS:
        t, p = tests[w]
        print(f'w{w}: Inc={np.mean(inc_folds[w]):.4f}±{np.std(inc_folds[w]):.4f}  '
              f'Tr={np.mean(tr_folds[w]):.4f}±{np.std(tr_folds[w]):.4f}  '
              f't={t:.3f}  p={p:.4f}  {_sig(p)}')
    plt.show()


if __name__ == '__main__':
    os.makedirs(FIGS, exist_ok=True)

    r_cnn_mb   = _load('cnn_multibranch_es_results.json')
    r_mb       = _load('inception_multibranch_es_results.json')
    r_tr_mb    = _load('transformer_multibranch_es_results.json')
    r_mamba_mb = _load('mamba_multibranch_es_results.json')

    fig1_frame_length_arch(r_cnn_mb, r_mb, r_tr_mb, r_mamba_mb)
    fig2_cv_comparison()
