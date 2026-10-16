"""Publication-style composites of existing paired evaluations; no GPU/API use.

Run: python experiment/scripts/plot_figure_summary.py
Writes PNG, editable SVG, numeric source data, and captions to figure_sum/.
"""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/material-experiment-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODES = ('base', 'tuned')
NAMES = ('Base', 'LoRA')
COLORS = ('#0072B2', '#D55E00')  # Okabe-Ito: blue / vermilion
INK, MUTED, GRID = '#19252F', '#596875', '#DDE3E8'
SEED, RESAMPLES = 20261014, 10000
TOPICS = {
    '工艺与施工': 'Processing',
    '氧化与侵蚀': 'Oxidation / corrosion',
    '热震与力学': 'Thermomechanics',
    '结构与表征/其他': 'Structure / other',
}
PROBE_GROUPS = {'提取': 'Extraction', '计算': 'Calculation', '工具路由': 'Tool routing', '证据边界': 'Evidence limits'}
FILES = (
    'results/summary.json', 'results/scored_generation.json',
    'results/full_prompt_nll/nll.json', 'results/comparison/timing.json',
    'results/comparison/manifest.json',
)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def paired(rows, kind=None):
    if kind is not None:
        rows = [r for r in rows if r['kind'] == kind]
    result = {mode: {} for mode in MODES}
    for row in rows:
        group = result[row['mode']]
        if row['case_id'] in group:
            raise ValueError(f'Duplicate {row["mode"]}/{row["case_id"]}')
        group[row['case_id']] = row
    if result['base'].keys() != result['tuned'].keys():
        raise ValueError('Unpaired before/after samples')
    return result


def mean_ci(before, after, weights=None, exponentiate=False):
    """Percentile paired bootstrap over cases, not tokens or training runs."""
    values = np.asarray([before, after], dtype=float)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError('Expected finite, nonempty paired vectors')
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, values.shape[1], (RESAMPLES, values.shape[1]))
    if weights is None:
        estimates = values[:, indices].mean(axis=2)
        centers = values.mean(axis=1)
    else:
        weights = np.asarray(weights, dtype=float)
        estimates = (values[:, indices] * weights[indices]).sum(axis=2) / weights[indices].sum(axis=1)
        centers = np.average(values, axis=1, weights=weights)
    if exponentiate:
        estimates, centers = np.exp(estimates), np.exp(centers)
    return {
        'n': values.shape[1],
        'mean': centers.tolist(),
        'ci95': np.quantile(estimates, [.025, .975], axis=1).T.tolist(),
        'delta_after_minus_before': float(centers[1] - centers[0]),
        'delta_ci95': np.quantile(estimates[1] - estimates[0], [.025, .975]).tolist(),
    }


def wilson(success, n):
    z = 1.959963984540054
    p, den = success / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [100 * max(0, center - half), 100 * min(1, center + half)]


def rate_stats(success, n):
    return {'n': n, 'counts': success, 'mean': [100 * k / n for k in success], 'ci95': [wilson(k, n) for k in success]}


def prepare(root=ROOT):
    summary = read(root / FILES[0])
    gen = read(root / FILES[1])
    qa, probes = paired(gen, 'qa'), paired(gen, 'probes')
    nll = paired(read(root / FILES[2]))
    batches = read(root / FILES[3])
    bad_ids = set(summary['data_audit']['empty_reference_ids'])
    ids = sorted(k for k in nll['base'] if k not in bad_ids)
    qa_ids = sorted(qa['base'])
    lexical_ids = [k for k in qa_ids if not qa['base'][k]['empty_reference']]
    if (len(ids), len(qa_ids), len(lexical_ids), len(probes['base'])) != (95, 24, 23, 16):
        raise ValueError('Unexpected sample counts; inspect the input run before plotting')
    for k in ids:
        a, b = (nll[m][k] for m in MODES)
        if a['scored_tokens'] != b['scored_tokens']:
            raise ValueError(f'Unequal scored-token counts: {k}')
        for row in (a, b):
            if row['sequence_tokens'] - row['scored_tokens'] != row['source_tokens_full']:
                raise ValueError('Expected full-prompt NLL evaluation')
    for k in qa_ids:
        a, b = (qa[m][k] for m in MODES)
        if (a['question'], a['reference'], a['group']) != (b['question'], b['reference'], b['group']):
            raise ValueError(f'Mismatched paired QA: {k}')
    nll_values = [[nll[m][k]['nll'] for k in ids] for m in MODES]
    weights = [nll['base'][k]['scored_tokens'] for k in ids]
    result = {
        'nll': mean_ci(*nll_values),
        'ppl': mean_ci(*nll_values, weights=weights, exponentiate=True),
        'nll_points': [{'case_id': k, 'group': nll['base'][k]['group'], 'before': a, 'after': b}
                       for k, a, b in zip(ids, *nll_values)],
        'nll_topics': {}, 'lexical': {}, 'lexical_topics': {}, 'instruction': {}, 'instruction_groups': {},
        'qa_case_ids': qa_ids, 'lexical_case_ids': lexical_ids,
        'probe_case_ids': sorted(probes['base']),
        'probe_exact_pass': [[probes[m][k]['exact_pass'] for k in sorted(probes['base'])] for m in MODES],
        'probe_valid_json': [[probes[m][k]['valid_json'] for k in sorted(probes['base'])] for m in MODES],
        'generation_lengths': [[qa[m][k]['new_tokens'] for k in qa_ids] for m in MODES],
        'cap_rate': rate_stats([sum(qa[m][k]['hit_token_limit'] for k in qa_ids) for m in MODES], len(qa_ids)),
        'timing': {},
    }
    for topic in TOPICS:
        selected = [k for k in ids if nll['base'][k]['group'] == topic]
        result['nll_topics'][topic] = mean_ci(*[[nll[m][k]['nll'] for k in selected] for m in MODES])
    for metric in ('char_rouge_l_f1', 'char_bigram_f1'):
        result['lexical'][metric] = mean_ci(*[[100 * qa[m][k][metric] for k in lexical_ids] for m in MODES])
        result['lexical_topics'][metric] = {}
        for topic in TOPICS:
            selected = [k for k in lexical_ids if qa['base'][k]['group'] == topic]
            result['lexical_topics'][metric][topic] = mean_ci(*[[100 * qa[m][k][metric] for k in selected] for m in MODES])
    result['lexical_deltas'] = {
        metric: [100 * (qa['tuned'][k][metric] - qa['base'][k][metric]) for k in lexical_ids]
        for metric in ('char_rouge_l_f1', 'char_bigram_f1')
    }
    for metric in ('exact_pass', 'valid_json'):
        result['instruction'][metric] = rate_stats([sum(r[metric] for r in probes[m].values()) for m in MODES], 16)
    for group in PROBE_GROUPS:
        selected = [k for k in probes['base'] if probes['base'][k]['group'] == group]
        result['instruction_groups'][group] = rate_stats([sum(probes[m][k]['exact_pass'] for k in selected) for m in MODES], len(selected))
    for mode in MODES:
        selected = [r for r in batches if r['mode'] == mode and r['kind'] == 'qa']
        if len(selected) != 6 or any(r['count'] != 4 for r in selected):
            raise ValueError('Expected six QA batches of four prompts per model')
        seconds = [r['seconds'] for r in selected]
        result['timing'][mode] = {
            'batch_seconds': seconds, 'median_batch_seconds': float(np.median(seconds)),
            'total_seconds': sum(seconds),
            'aggregate_tokens_per_second': sum(r['new_tokens'] for r in selected) / sum(seconds),
        }
        expected = summary['qa_timing'][mode]
        for key in ('median_batch_seconds', 'aggregate_tokens_per_second'):
            if not math.isclose(result['timing'][mode][key], expected[key], rel_tol=1e-10):
                raise ValueError(f'Timing changed: {mode}/{key}')
    checks = [('nll', 'base_mean', 'tuned_mean'), ('ppl', 'base_token_weighted_ppl', 'tuned_token_weighted_ppl')]
    for metric, before_key, after_key in checks:
        for observed, key in zip(result[metric]['mean'], (before_key, after_key)):
            if not math.isclose(observed, summary['nll_full_prompt'][key], rel_tol=1e-10):
                raise ValueError(f'Published summary mismatch: {metric}/{key}')
    for i, mode in enumerate(MODES):
        for metric in result['lexical']:
            if not math.isclose(result['lexical'][metric]['mean'][i], 100 * summary['qa'][mode][metric], rel_tol=1e-10):
                raise ValueError('Published lexical score mismatch')
    return result


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'axes.labelsize': 10.5, 'axes.titlesize': 11.5, 'axes.titleweight': 'semibold',
        'axes.titlepad': 13, 'axes.edgecolor': '#74818B', 'axes.linewidth': .7,
        'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'svg.fonttype': 'none', 'svg.hashsalt': 'material-summary-20261016',
        'savefig.facecolor': 'white', 'axes.axisbelow': True,
    })


def canvas(title, subtitle, rows=2, cols=2, size=(12.8, 8.5), legend=True):
    fig = plt.figure(figsize=size)
    grid = fig.add_gridspec(rows, cols, left=.09, right=.96, bottom=.17, top=.79, hspace=.62,
                           wspace=.55 if cols == 2 else .40)
    fig.text(.045, .95, title, fontsize=18, weight='semibold', va='top')
    fig.text(.045, .903, subtitle, fontsize=10, color=MUTED, va='top')
    if legend:
        fig.legend([Patch(facecolor=c) for c in COLORS], ['Base Qwen3-8B', 'LoRA checkpoint-600'],
                   loc='upper left', bbox_to_anchor=(.04, .876), ncol=2, frameon=False, fontsize=10,
                   handlelength=1.2, columnspacing=2.2)
    return fig, grid


def panel(ax, letter, title, axis='y'):
    ax.set_title(title, loc='left')
    label_x = -.05 if ax.get_position().width > .6 else -.12
    ax.text(label_x, 1.09, letter, transform=ax.transAxes, weight='bold', fontsize=14, va='bottom')
    ax.grid(axis=axis, color=GRID, lw=.65)
    ax.tick_params(length=3, width=.6)


def footnote(fig, lines):
    for i, line in enumerate(lines):
        fig.text(.045, .098 - i * .027, line, fontsize=8.7, color=MUTED, va='top')


def pair_bars(ax, values, unit, ci=None, digits=3, ymax=None):
    ax.bar([0, 1], values, color=COLORS, width=.55, alpha=.92)
    top = max(max(v) for v in ci) if ci is not None else max(values)
    ax.set_ylim(0, ymax if ymax is not None else top * 1.30)
    if ci is not None:
        for i, (value, interval) in enumerate(zip(values, ci)):
            ax.errorbar(i, value, yerr=np.array([[value - interval[0]], [interval[1] - value]]),
                        fmt='none', ecolor=INK, capsize=4, lw=1.05)
    for i, value in enumerate(values):
        label_y = ci[i][1] if ci is not None else value
        ax.annotate(f'{value:.{digits}f}', (i, label_y), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=10.5, weight='semibold')
    ax.set_xticks([0, 1], NAMES)
    ax.set_ylabel(unit)
    ax.set_xlim(-.65, 1.65)


def grouped_bars(ax, labels, stats, ylabel, percent=False, ci=True):
    xs = np.arange(len(labels))
    max_value = 0
    for i, color in enumerate(COLORS):
        centers = [s['mean'][i] for s in stats]
        pos = xs + (i - .5) * .34
        ax.bar(pos, centers, width=.30, color=color, alpha=.92)
        for x, value, s in zip(pos, centers, stats):
            interval = s['ci95'][i]
            if ci:
                ax.errorbar(x, value, yerr=[[max(0, value - interval[0])], [max(0, interval[1] - value)]],
                            fmt='none', color=INK, capsize=3, lw=.85)
            y = interval[1] if ci else value
            max_value = max(max_value, y)
            label = f'{s["counts"][i]}/{s["n"]}' if percent else f'{value:.2f}'
            ax.annotate(label, (x, y), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9, weight='semibold')
    ax.set_xticks(xs, labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 119 if percent else max_value * 1.25)
    if percent:
        ax.set_yticks([0, 25, 50, 75, 100])


def topic_points(ax, stats, xlabel, score=False):
    for j, (topic, row) in enumerate(stats.items()):
        ax.plot(row['mean'], [j, j], color='#A7B1BA', lw=1.25)
        for i in range(2):
            value, interval = row['mean'][i], row['ci95'][i]
            ax.errorbar(value, j + (i - .5) * .14,
                        xerr=[[max(0, value - interval[0])], [max(0, interval[1] - value)]],
                        fmt='o' if i == 0 else 's', color=COLORS[i], markersize=5,
                        capsize=3, lw=1.1, markeredgecolor='white', markeredgewidth=.5)
    ax.set_yticks(range(len(stats)), [f'{TOPICS[k]}  (n={v["n"]})' for k, v in stats.items()], fontsize=9)
    ax.invert_yaxis()
    ax.set_ylim(len(stats) - .5, -.5)
    ax.set_xlim((0, 45) if score else (0.8, 1.95))
    ax.set_xlabel(xlabel)


def write_figure(fig, out, name):
    png = out / f'{name}.png'
    fig.savefig(png, dpi=300, metadata={'Software': f'Matplotlib {matplotlib.__version__}; paired measured data'})
    fig.savefig(out / f'{name}.svg', metadata={'Date': '2026-10-16', 'Creator': 'plot_figure_summary.py'})
    plt.close(fig)
    print(png)


def likelihood_figure(data, out):
    fig, grid = canvas('Reference-answer likelihood', 'Qwen3-8B material-domain adaptation | full prompts retained; reference answers may be truncated')
    ax = fig.add_subplot(grid[0, 0]); panel(ax, 'a', 'Sample-mean NLL  (lower is better)')
    pair_bars(ax, data['nll']['mean'], 'Mean response-token NLL', data['nll']['ci95'])
    ax = fig.add_subplot(grid[0, 1]); panel(ax, 'b', 'Token-weighted perplexity  (lower is better)')
    pair_bars(ax, data['ppl']['mean'], 'Perplexity (PPL)', data['ppl']['ci95'])
    ax = fig.add_subplot(grid[1, 0]); panel(ax, 'c', 'Paired NLL by question', axis='both')
    before = np.array([p['before'] for p in data['nll_points']]); after = np.array([p['after'] for p in data['nll_points']])
    low, high = min(before.min(), after.min()) - .1, max(before.max(), after.max()) + .1
    ax.plot([low, high], [low, high], '--', color=MUTED, lw=1)
    ax.scatter(before, after, s=23, color=COLORS[1], alpha=.65, linewidths=.3, edgecolors='white')
    ax.set(xlim=(low, high), ylim=(low, high), xlabel='Base NLL', ylabel='LoRA NLL')
    ax.text(.04, .94, f'{int((after < before).sum())}/{len(before)} lower after tuning', transform=ax.transAxes, va='top', fontsize=10)
    ax = fig.add_subplot(grid[1, 1]); panel(ax, 'd', 'Domain-stratified NLL', axis='x')
    topic_points(ax, data['nll_topics'], 'Mean NLL (lower is better)')
    footnote(fig, [
        'n=95 nonempty references; one empty reference excluded. NLL is averaged across cases; PPL is weighted by scored reference tokens.',
        'Error bars: 95% paired percentile bootstrap intervals (10,000 case resamples). Same tokenizer and target tokens in each pair.',
        'Validation was used for checkpoint selection; not an independent test. Intervals do not correct selection bias or document clustering.',
    ])
    write_figure(fig, out, '01_reference_likelihood')


def lexical_figure(data, out):
    fig, grid = canvas('Generated-answer lexical agreement', 'Character-level reference overlap | scores are not factual accuracy or expert quality ratings')
    ax = fig.add_subplot(grid[0, 0]); panel(ax, 'a', 'Overall overlap  (higher is better)')
    grouped_bars(ax, ['ROUGE-L F1', 'Bigram F1'], list(data['lexical'].values()), 'Character-level F1 × 100')
    ax = fig.add_subplot(grid[0, 1]); panel(ax, 'b', 'Domain-stratified ROUGE-L', axis='x')
    topic_points(ax, data['lexical_topics']['char_rouge_l_f1'], 'Character ROUGE-L F1 × 100', score=True)
    ax = fig.add_subplot(grid[1, 0]); panel(ax, 'c', 'Per-question changes', axis='x')
    for i, (metric, stats) in enumerate(data['lexical'].items()):
        values = np.array(data['lexical_deltas'][metric])
        jitter = np.random.default_rng(SEED + i).uniform(-.13, .13, len(values))
        ax.scatter(values, i + jitter, color='#657786', s=22, alpha=.7, edgecolors='white', linewidths=.4)
        center, interval = stats['delta_after_minus_before'], stats['delta_ci95']
        ax.errorbar(center, i - .28, xerr=[[center - interval[0]], [interval[1] - center]], fmt='D', color=INK, capsize=4, markersize=5)
    ax.axvline(0, ls='--', color=MUTED, lw=1)
    ax.set(yticks=[0, 1], yticklabels=['ROUGE-L F1', 'Bigram F1'], xlabel='LoRA − Base (score points)', ylim=(-.55, 1.45))
    ax.text(.03, .94, 'Points: individual QA; diamonds: mean (95% CI)', fontsize=8.3, color=MUTED, transform=ax.transAxes, va='top')
    ax = fig.add_subplot(grid[1, 1]); panel(ax, 'd', 'Domain-stratified bigram F1', axis='x')
    topic_points(ax, data['lexical_topics']['char_bigram_f1'], 'Character bigram F1 × 100', score=True)
    footnote(fig, [
        'n=23 nonempty-reference QA pairs; macro-average F1 after casefolding and retaining Unicode letters/numbers only.',
        '95% paired bootstrap intervals; small topic subsets (n=5–6) are descriptive. Reference answers have not been independently verified.',
        'Generation capped at 512 tokens: Base 24/24 and LoRA 19/24 hit the cap. Length limits can affect overlap scores.',
    ])
    write_figure(fig, out, '02_answer_overlap')


def instruction_figure(data, out):
    fig, grid = canvas('Instruction and output-format diagnostics', '16 hand-authored paired probes | deterministic JSON parsing and exact expected-value checks')
    ax = fig.add_subplot(grid[0, 0]); panel(ax, 'a', 'Overall pass rates  (higher is better)')
    grouped_bars(ax, ['Strict instruction', 'Valid JSON'], list(data['instruction'].values()), 'Pass rate (%)', percent=True)
    ax = fig.add_subplot(grid[0, 1]); panel(ax, 'b', 'Strict pass rate by probe category')
    grouped_bars(ax, ['Extraction', 'Calculation', 'Tool\nrouting', 'Evidence\nlimits'], list(data['instruction_groups'].values()), 'Pass rate (%)', percent=True)
    ax = fig.add_subplot(grid[1, :]); panel(ax, 'c', 'Paired case-level outcomes', axis='x')
    ax.set_title('Paired case-level outcomes', loc='left', pad=40)
    ax.texts[0].set_y(1.26)
    passed = np.array(data['probe_exact_pass'], dtype=int)
    ax.grid(False)
    ax.imshow(passed, cmap=ListedColormap(['#F1E7DD', '#1D6E85']), vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    for i in range(2):
        for j in range(16):
            ax.text(j, i, '1' if passed[i, j] else '0', ha='center', va='center', color='white' if passed[i, j] else INK, fontsize=11)
            if not data['probe_valid_json'][i][j]:
                ax.plot(j, i + .29, marker='x', color=COLORS[1], ms=8, mew=2)
    for boundary in [3.5, 7.5, 11.5]: ax.axvline(boundary, color='white', lw=4)
    for j, label in enumerate(PROBE_GROUPS.values()):
        ax.text(j * 4 + 1.5, -.58, label, ha='center', va='bottom', fontsize=10, weight='semibold')
    ax.set_xticks(range(16), [k.replace('probe-', '') for k in data['probe_case_ids']])
    ax.set_yticks([0, 1], NAMES)
    ax.set_xlabel('Probe ID  |  1 = strict pass; 0 = fail; × = invalid raw JSON')
    ax.set_ylim(1.5, -.5)
    for spine in ax.spines.values(): spine.set_visible(False)
    footnote(fig, [
        'Error bars: 95% Wilson intervals; these small authored subsets are descriptive, not representative population estimates.',
        'Strict pass requires valid JSON and matching keys/values/types. Numeric strings and Markdown fences can cause failure.',
        'Probe 08 improves; probe 05 regresses. Tool-routing tasks do not execute tools. These are not IFEval or BFCL benchmark scores.',
    ])
    write_figure(fig, out, '03_instruction_format')


def efficiency_figure(data, out):
    fig, grid = canvas('Generation efficiency and output limits', 'Tesla V100S 32 GB | FP16 · SDPA · batch=4 · greedy · thinking disabled · unmerged LoRA')
    throughput = [data['timing'][m]['aggregate_tokens_per_second'] for m in MODES]
    ax = fig.add_subplot(grid[0, 0]); panel(ax, 'a', 'Aggregate output throughput  (higher is better)')
    pair_bars(ax, throughput, 'Generated tokens / second', digits=2)
    ax = fig.add_subplot(grid[0, 1]); panel(ax, 'b', 'Observed batch durations  (lower is better)')
    for i, mode in enumerate(MODES):
        values = data['timing'][mode]['batch_seconds']
        ax.plot(range(1, 7), values, marker='o' if i == 0 else 's', color=COLORS[i], lw=1.4, ms=5)
        median = data['timing'][mode]['median_batch_seconds']
        ax.axhline(median, color=COLORS[i], ls=':', lw=1, alpha=.8)
        ax.text(.03, .95 - i * .12, f'{NAMES[i]} median: {median:.2f} s / batch', transform=ax.transAxes, color=COLORS[i], fontsize=9.5, va='top')
    ax.set(xlabel='Batch index (4 prompts / batch)', ylabel='Seconds / batch', xticks=range(1, 7), ylim=(0, 66))
    ax = fig.add_subplot(grid[1, 0]); panel(ax, 'c', 'Output length distribution')
    lengths = np.array(data['generation_lengths'])
    offsets = np.linspace(-.19, .19, lengths.shape[1])
    for j in range(lengths.shape[1]):
        ax.plot([0 + offsets[j], 1 + offsets[j]], lengths[:, j], color='#AFB8C0', lw=.65, alpha=.45)
    for i, color in enumerate(COLORS):
        ax.scatter(i + offsets, lengths[i], color=color, s=24, alpha=.8, edgecolors='white', linewidths=.4, zorder=3)
        value = lengths[i].mean()
        ax.plot([i - .25, i + .25], [value, value], color=INK, lw=2, zorder=4)
        ax.text(i, 563, f'Mean {value:.2f}', ha='center', fontsize=10, weight='semibold')
    ax.axhline(512, ls='--', color=MUTED, lw=1)
    ax.set(xticks=[0, 1], xticklabels=NAMES, ylabel='Generated tokens / answer', ylim=(0, 600), xlim=(-.6, 1.6))
    ax = fig.add_subplot(grid[1, 1]); panel(ax, 'd', 'Outputs reaching the 512-token cap')
    pair_bars(ax, data['cap_rate']['mean'], 'Cap-hit rate (%)', digits=2, ymax=120)
    ax.set_yticks([0, 25, 50, 75, 100])
    for i, count in enumerate(data['cap_rate']['counts']):
        ax.text(i, 8, f'{count}/24', color='white', ha='center', weight='bold')
    footnote(fig, [
        '24 identical QA prompts per model, six batches each; warm-up excluded. Throughput = total output tokens / total generation time.',
        'Batch duration is not per-request latency or time-to-first-token. One sequential run; no serving-load or repeated-run uncertainty study.',
        'Adapters were not merged. Output length has no monotonic quality preference; hitting the token cap may leave an answer incomplete.',
    ])
    write_figure(fig, out, '04_generation_efficiency')


def overview_figure(data, out):
    fig, grid = canvas('Qwen3-8B: paired fine-tuning evaluation', 'Measured results only | Base model versus selected LoRA checkpoint-600 | evaluation: 14 October 2026',
                       rows=2, cols=3, size=(15.2, 8.4))
    ax = fig.add_subplot(grid[0, 0]); panel(ax, 'a', 'Reference NLL ↓  (n=95)')
    pair_bars(ax, data['nll']['mean'], 'Mean NLL', data['nll']['ci95'])
    ax = fig.add_subplot(grid[0, 1]); panel(ax, 'b', 'Reference perplexity ↓  (n=95)')
    pair_bars(ax, data['ppl']['mean'], 'Token-weighted PPL', data['ppl']['ci95'])
    ax = fig.add_subplot(grid[0, 2]); panel(ax, 'c', 'Lexical agreement ↑  (n=23)')
    grouped_bars(ax, ['ROUGE-L', 'Bigram F1'], list(data['lexical'].values()), 'Character F1 × 100')
    ax = fig.add_subplot(grid[1, 0]); panel(ax, 'd', 'Instruction / format ↑  (n=16)')
    grouped_bars(ax, ['Strict pass', 'Valid JSON'], list(data['instruction'].values()), 'Pass rate (%)', percent=True)
    ax = fig.add_subplot(grid[1, 1]); panel(ax, 'e', 'Output throughput ↑')
    pair_bars(ax, [data['timing'][m]['aggregate_tokens_per_second'] for m in MODES], 'Generated tokens / second', digits=2)
    ax = fig.add_subplot(grid[1, 2]); panel(ax, 'f', 'Median batch duration ↓')
    pair_bars(ax, [data['timing'][m]['median_batch_seconds'] for m in MODES], 'Seconds / batch of 4', digits=2)
    footnote(fig, [
        'Panels a–c: 95% paired bootstrap intervals; panel d: 95% Wilson intervals. Panels e–f: one sequential run, six QA batches per model.',
        'Validation participated in checkpoint selection. Lexical overlap is not factual accuracy; 24/24 Base and 19/24 LoRA answers reached the 512-token cap.',
        'V100S · FP16 · SDPA · greedy · thinking disabled · unmerged LoRA. Hand-authored probes are not standard IFEval/BFCL scores. No unmeasured scores shown.',
    ])
    write_figure(fig, out, '00_overview')


CAPTIONS = {
    '00_overview': ('核心结果总览', '6个子图汇总NLL、PPL、文本重合、指令/格式通过率及吞吐/批次耗时。不同量纲分别作图，不合成为总分，也不使用雷达图混合尺度。'),
    '01_reference_likelihood': ('参考答案拟合与领域分组', '95条非空参考，保留完整prompt；分别展示逐样本等权NLL、token加权PPL、同题散点及4个主题分组。它们衡量参考答案似然，不是生成答案事实正确率。'),
    '02_answer_overlap': ('材料回答文本重合度', '23组非空参考问答的字符ROUGE-L与二元组F1，以及主题分组和逐题差值。新增主题二元组分组由已有逐题分数聚合，未重新进行GPU推理。负向变化不隐藏。'),
    '03_instruction_format': ('指令遵循与JSON格式', '16组自编题：总体通过率、4类子任务及逐题成对结果。柱上标成功数/总数，误差线为Wilson区间。数字字符串或Markdown围栏可能导致严格评分失败，不能直接解释为计算错误。'),
    '04_generation_efficiency': ('生成效率与长度限制', '同一V100S环境下的输出吞吐、6批实际耗时、24题输出长度及512-token触顶率。LoRA未合并；批次耗时不是单请求延迟，未测TTFT或P95请求延迟。'),
}


def supporting_files(data, out, root):
    (out / 'source_data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    manifest = {
        'source_evaluation_date': '2026-10-14', 'figure_prepared_date': '2026-10-16',
        'operation': 'Plot existing measured results only; no new inference, training, or external upload.',
        'source_sha256': {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in FILES},
        'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'versions': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'bootstrap': {'paired': True, 'unit': 'question', 'resamples': RESAMPLES, 'seed': SEED, 'interval': 'percentile 95%'},
        'proportion_intervals': 'Wilson score 95%; descriptive for small hand-authored probe subsets',
        'limitations': read(root / 'results/comparison/manifest.json')['limitations'],
        'nll_protocol': 'Full prompt retained; reference response truncated to fit cutoff=1536; n=95 nonempty references.',
        'formats': ['300-DPI PNG', 'SVG with editable text'],
        'figures': {name: {'title_zh': title, 'caption_zh': caption} for name, (title, caption) in CAPTIONS.items()},
    }
    (out / 'figure_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    notes = ['微调前后已测指标：学术组合图', '',
             '所有图片源于2026-10-14的真实评测；本次只绘图，无新训练/推理，无外部上传。',
             'PNG为300 DPI；SVG保留矢量元素及可编辑文字。图内英文，以下提供中文图注。',
             '颜色固定：蓝色Base、橙色LoRA checkpoint-600；图03逐题矩阵的深色/浅色表示通过/失败。', '',
             *[f'{name}.png / .svg\n{title}：{caption}\n' for name, (title, caption) in CAPTIONS.items()],
             '统计口径与限制',
             '1. 连续指标为10000次按问题配对重采样的95% percentile bootstrap区间，seed=20261014。',
             '2. PPL为exp(token加权NLL)，不等于exp(逐样本等权平均NLL)。',
             '3. 指令通过率区间为95% Wilson区间，不校正自编题便利抽样偏差，不代表总体性能。',
             '4. 所有区间条件于已选checkpoint，未校正选模偏差、文档相关性或多重比较；没有多训练种子复现实验。',
             '5. 1条空参考不计入NLL/重合分数，但保留在24题长度和耗时统计中。',
             '6. QA输出截断影响重合分数；参考答案未经专家盲评，不能把ROUGE/F1当事实准确率。',
             '7. 性能为单次顺序评测，吞吐按总输出token/总生成时间；未测单请求P95、TTFT或峰值显存。',
             '8. 未测试的标准benchmark没有绘制，也没有用官方基模分数代替本项目评测。', '',
             '复现（项目根目录，环境需numpy和matplotlib）',
             'python experiment/scripts/plot_figure_summary.py',
             'python -m unittest discover -s experiment/scripts -p test_figure_summary.py -v', '',
             'source_data.json：所有绘图数值及区间；figure_manifest.json：原始文件SHA256、方法和软件版本。',
             '原始experiment/results及原图experiment/figures不作修改。']
    (out / 'README.txt').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    cards = '\n'.join(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p><a href="{name}.png"><img src="{name}.png" alt="{html.escape(title)}"></a><p><a href="{name}.png">PNG 原图</a> · <a href="{name}.svg">SVG 矢量图</a></p></section>' for name, (title, caption) in CAPTIONS.items())
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>微调前后：已测指标组合图</title><style>body{max-width:1400px;margin:36px auto;padding:0 24px;font:16px/1.7 system-ui;color:#19252f;background:#f4f6f8}section{background:white;padding:20px;margin:24px 0;border:1px solid #dde3e8;border-radius:8px}img{width:100%;height:auto}h1{font-size:28px}h2{font-size:21px}a{color:#0072b2}p{color:#596875}</style><h1>Qwen3-8B 微调前后：已测指标组合图</h1><p>源评测：2026-10-14 · 蓝色为原始模型，橙色为LoRA checkpoint-600。没有重新运行训练或推理。</p><p>验证集参与过选模，非独立测试；文本重合度不代表事实准确率。<a href="README.txt">完整图注与统计限制</a> · <a href="source_data.json">绘图数据</a></p>' + cards + '</html>'
    (out / 'index.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'figure_sum')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = prepare()
    style()
    for plot in (likelihood_figure, lexical_figure, instruction_figure, efficiency_figure, overview_figure):
        plot(data, args.output)
    supporting_files(data, args.output, ROOT)


if __name__ == '__main__':
    main()
