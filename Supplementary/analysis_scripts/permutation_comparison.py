"""permutation comparison. See Supplementary/README.md."""
import os

def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)

from pathlib import Path
import hashlib, json, platform
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import sklearn

ROOT = Path(os.environ.get("ORIGENE_BENCHMARK_ROOT", "Data/Main_figures/benchmark")).expanduser().resolve()
CSV_PATH = ROOT / 'final_prediction_merged_all.csv'
OUT = Path("results/supplementary/permutation_comparison")
B_PERMS = 5000
SEED = 42
TARGET = 'OriGENE'  # Manuscript's 4-panel permutation figure (not OriGENE_Mut)
BASELINES = ['20/20+', 'DORGE_mut_ElasticNet', 'DORGE_epi_ElasticNet', 'DORGE_all_ElasticNet']
MODELS = ['OriGENE','OriGENE_Mut'] + BASELINES  # Match original _wide() selection exactly
LABELS = {'OriGENE':'OriGENE (Epi)', '20/20+':'20/20+ (Mut)',
          'DORGE_mut_ElasticNet':'DORGE (Mut)',
          'DORGE_epi_ElasticNet':'DORGE (Epi)',
          'DORGE_all_ElasticNet':'DORGE (Epi + Mut)'}
CACHE = OUT / 'legacy_null_differences.npz'
assert CSV_PATH.is_file(), f'Missing input CSV: {CSV_PATH}'
OUT.mkdir(parents=True, exist_ok=True)
print('Input:', CSV_PATH)
print('Output (new folder; original results untouched):', OUT)
print('Legacy cache exists:', CACHE.exists())

raw = pd.read_csv(CSV_PATH)
required = {'gene', 'model', 'dataset', 'label', 'score'}
assert required.issubset(raw.columns), required - set(raw.columns)
raw = raw.loc[raw['model'].isin(MODELS)].copy()
assert raw['score'].notna().all() and np.isfinite(raw['score']).all()
assert set(raw['label'].unique()).issubset({0,1})
assert not raw.duplicated(['gene','model','dataset']).any(), 'Repeated gene/model/dataset rows: provenance must be checked first.'
assert raw.groupby('gene')['label'].nunique().max() == 1, 'Conflicting class labels.'

# Reproduce premutation_test.ipynb Cell 28's _wide():
W = raw.groupby(['gene','model'])['score'].mean().unstack()
W = W[[m for m in W.columns if m in MODELS]]
W = W.join(raw.groupby('gene')['label'].first().rename('TRUE'), how='inner').dropna()
y = W['TRUE'].to_numpy(dtype=int)
assert set(MODELS).issubset(W.columns) and set(np.unique(y)) == {0,1}
counts = raw.groupby(['gene','model'])['dataset'].nunique().unstack()
common_coverage = counts.reindex(index=W.index, columns=MODELS)
print('Legacy common genes:', len(W), '| CDs:', y.sum(), '| NGs:', len(y)-y.sum())
print('Datasets per common gene/model (min/max):', int(common_coverage.min().min()), int(common_coverage.max().max()))
if common_coverage.min().min() != common_coverage.max().max():
    print('WARNING: Unequal gene/model dataset coverage; legacy averaging weights models differently.')

obs = np.array([roc_auc_score(y, W[TARGET]) - roc_auc_score(y, W[b]) for b in BASELINES])
for b, d in zip(BASELINES, obs):
    print(f'{LABELS[TARGET]} – {LABELS[b]}: observed delta AUROC = {d:.6f}')

raw_sha = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
print('CSV SHA256:', raw_sha)

def legacy_null_pair(y, a, b, n_perm=5000, seed=42):
    # Exact algorithm of the original premutation_test.ipynb Cell 28
    rng = np.random.default_rng(seed)
    n = len(y)
    swaps = rng.random((n_perm, n)) > 0.5
    diffs = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        a_new = np.where(swaps[i], b, a)
        b_new = np.where(swaps[i], a, b)
        diffs[i] = roc_auc_score(y, a_new) - roc_auc_score(y, b_new)
    return diffs

if CACHE.exists():
    saved = np.load(CACHE, allow_pickle=False)
    if (str(saved['csv_sha256']) != raw_sha or
        int(saved['seed']) != SEED or
        int(saved['B']) != B_PERMS or
        str(saved['target']) != TARGET or
        not np.array_equal(saved['baselines'], np.array(BASELINES))):
        raise ValueError('Cache provenance differs from current input/configuration. Do not reuse.')
    null = saved['null_differences']  # (4, B)
    print('Loaded existing ORIGINAL-method permutations from:', CACHE)
else:
    print('Cache absent: recomputing the ORIGINAL legacy paired-swap permutations.')
    null = np.vstack([
        legacy_null_pair(y, W[TARGET].to_numpy(), W[b].to_numpy(), B_PERMS, SEED)
        for b in BASELINES
    ])
    np.savez_compressed(CACHE, null_differences=null,
                        csv_sha256=raw_sha, seed=SEED, B=B_PERMS,
                        target=TARGET, baselines=np.array(BASELINES))
    print('Cached ORIGINAL-method permutations for future plot-only redraw:', CACHE)
assert null.shape == (4, B_PERMS)

# Original formulas: two-sided +1 correction; 4-comparison single-step maxT
p_perm = (1 + (np.abs(null) >= np.abs(obs[:,None])).sum(axis=1)) / (B_PERMS+1)
max_abs = np.max(np.abs(null), axis=0)
p_maxT = (1 + (max_abs[None,:] >= np.abs(obs[:,None])).sum(axis=1)) / (B_PERMS+1)
assert (p_maxT + 1e-12 >= p_perm).all()
assert ((p_perm > 0) & (p_maxT > 0) & (p_perm <= 1) & (p_maxT <= 1)).all()

result = pd.DataFrame({'panel':['a','b','c','d'],
                       'target':[TARGET]*4, 'baseline':BASELINES,
                       'delta_AUROC':obs, 'p_perm':p_perm, 'p_maxT_4':p_maxT})
display(result)
result.to_csv(OUT / 'S9_legacy_four_comparison_maxT_results.csv', index=False)
meta = {'estimator':'AUROC of scores averaged per gene and model over datasets',
        'multiple_testing':'legacy synchronized paired-swap; single-step maxT across 4 baselines',
        'input_csv_sha256':raw_sha, 'n_common_genes':len(W),
        'n_CDs':int(y.sum()), 'n_NGs':int(len(y)-y.sum()),
        'B':B_PERMS, 'seed':SEED, 'target':TARGET, 'baselines':BASELINES,
        'sklearn_version':sklearn.__version__, 'python':platform.python_version(),
        'notes':['Exact legacy algorithm reproduced; no Holm; not pooled OOF.',
                 'Validity of joint resampling/subset pivotality not established by reproduction.']}
(OUT / 'S9_legacy_manifest.json').write_text(json.dumps(meta, indent=2))

from matplotlib.ticker import MaxNLocator, FormatStrFormatter

plt.rcParams.update({
    'font.family':'serif', 'font.serif':['Times New Roman','Times','DejaVu Serif'],
    'font.size':8, 'axes.labelsize':8, 'xtick.labelsize':7,
    'ytick.labelsize':7, 'axes.spines.top':False, 'axes.spines.right':False,
    'svg.fonttype':'none', 'savefig.facecolor':'white'
})
fig, axes = plt.subplots(2, 2, figsize=(180/25.4, 123/25.4))
# Common axis limits built from *all* draws and observations: no hidden clipped null values
lo = min(0.0, float(null.min()), float(obs.min()))
hi = max(0.0, float(null.max()), float(obs.max()))
margin = 0.05 * (hi-lo)
limits = (lo-margin, hi+margin)
edges = np.linspace(*limits, 37)
max_count = max(np.histogram(row, bins=edges)[0].max() for row in null)
for k, ax in enumerate(axes.flat):
    ax.hist(null[k], bins=edges, color='#B9CDE2', edgecolor='#547DA8', linewidth=0.55)
    ax.axvline(0, color='#777777', linestyle='--', lw=1)
    ax.axvline(obs[k], color='#C45D32', lw=1.5)
    ax.text(0.02, 1.04, f'({"abcd"[k]})', transform=ax.transAxes,
            fontsize=9, weight='bold', ha='left', va='bottom')
    ax.text(0.98, 0.96,
            f'Observed ΔAUROC = {obs[k]:.3f}\n'
            f'$p_{{\\mathrm{{perm}}}}$ = {p_perm[k]:.3g}\n'
            f'$p_{{\\mathrm{{maxT}}}}$ = {p_maxT[k]:.3g}',
            transform=ax.transAxes, ha='right', va='top', fontsize=7,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.95, pad=2))
    ax.set(title=LABELS[BASELINES[k]], xlim=limits,
           ylim=(0,max_count*1.32), xlabel='ΔAUROC (OriGENE − baseline)', ylabel='Count')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(direction='out', length=3)
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.11,
                    top=0.90, wspace=0.25, hspace=0.53)
fig.savefig(OUT/'S9_maxT_original_method_replot_600dpi.png',dpi=600,facecolor='white')
# Additional export omitted; retain canonical PNG.
plt.show()
print('Saved outputs to', OUT)
