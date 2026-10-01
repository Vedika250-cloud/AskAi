"""
AskAI Evaluation Chart Generator
Reads data/eval_results.json and produces charts in data/eval_charts/.

Usage:
    python scripts/eval-charts.py

Requires: matplotlib (pip install matplotlib)
"""

import json
import os
import sys
from pathlib import Path

# ── Guard ─────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')           # headless / no display needed
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("ERROR: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
RESULTS_F  = ROOT / 'data' / 'eval_results.json'
CHARTS_DIR = ROOT / 'data' / 'eval_charts'
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

if not RESULTS_F.exists():
    print(f"ERROR: {RESULTS_F} not found. Run: node scripts/eval-retrieval.js first.")
    sys.exit(1)

with open(RESULTS_F, 'r', encoding='utf-8') as f:
    data = json.load(f)

summary   = data['summary']
per_topic = data['per_topic']
results   = data['results']

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'font.size':       10,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'axes.grid.axis':     'y',
    'grid.alpha':         0.35,
    'grid.linestyle':     '--',
    'figure.dpi':         150,
})

NEUTRAL  = '#1a1a1a'
GRAY1    = '#555555'
GRAY2    = '#aaaaaa'
ACCENT   = '#2563eb'  # blue-600
SUCCESS  = '#16a34a'  # green-600
FAIL     = '#dc2626'  # red-600
WARN     = '#d97706'  # amber-600
BG       = '#ffffff'

# ── Chart 1: Overall Summary Bar ──────────────────────────────────────────────
def chart_overall_summary():
    n       = summary['successful_embed']
    top_k   = summary['top_k']
    metrics = {
        f'Hit@{top_k}\n(expected doc\nin top {top_k})': summary['hit_at_k_rate_pct'],
        'Hit@1\n(expected doc\nis rank 1)':              summary['hit_at_1_rate_pct'],
        f'Retrieved\n(sim ≥ {summary["threshold"]})':   summary['retrieval_rate_pct'],
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        list(metrics.keys()),
        list(metrics.values()),
        width=0.5,
        color=[ACCENT, SUCCESS, WARN],
        zorder=3,
    )

    for bar, val in zip(bars, metrics.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f'{val:.1f}%',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=NEUTRAL,
        )

    ax.set_ylim(0, 115)
    ax.set_ylabel('Percentage (%)', color=GRAY1)
    ax.set_title(
        f'AskAI Retrieval Performance  |  n={n} questions',
        fontsize=12, fontweight='bold', color=NEUTRAL, pad=12,
    )
    ax.tick_params(colors=GRAY1)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    fig.tight_layout()

    out = CHARTS_DIR / 'chart1_overall_summary.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out.name}')

# ── Chart 2: Per-Topic Hit@K Rates ────────────────────────────────────────────
def chart_per_topic_hitk():
    topics  = [t['topic'].replace(' ', '\n') for t in per_topic]
    hit_pct = [t['hit_at_k_pct'] for t in per_topic]
    avg_sim = [t['avg_sim'] for t in per_topic]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # — Hit@K by topic —
    colors = [SUCCESS if v >= 70 else WARN if v >= 40 else FAIL for v in hit_pct]
    bars = ax1.bar(topics, hit_pct, color=colors, width=0.55, zorder=3)
    for bar, val in zip(bars, hit_pct):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f'{val:.0f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color=NEUTRAL,
        )
    ax1.set_ylim(0, 115)
    ax1.set_ylabel('Hit Rate (%)', color=GRAY1)
    ax1.set_title(f'Hit@{summary["top_k"]} Rate by Topic', fontweight='bold', color=NEUTRAL)
    ax1.tick_params(colors=GRAY1, labelsize=8)

    # — Avg top similarity by topic —
    bars2 = ax2.bar(topics, avg_sim, color=ACCENT, width=0.55, zorder=3)
    ax2.axhline(summary['threshold'], color=FAIL, linestyle='--', linewidth=1.2,
                label=f'Threshold ({summary["threshold"]})', zorder=4)
    for bar, val in zip(bars2, avg_sim):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=8.5, color=NEUTRAL,
        )
    ax2.set_ylim(0, max(avg_sim) * 1.25)
    ax2.set_ylabel('Avg Top-1 Similarity', color=GRAY1)
    ax2.set_title('Average Similarity Score by Topic', fontweight='bold', color=NEUTRAL)
    ax2.tick_params(colors=GRAY1, labelsize=8)
    ax2.legend(fontsize=8)

    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    fig.suptitle('Per-Topic Retrieval Breakdown', fontsize=11, color=GRAY1, y=1.02)
    fig.tight_layout()

    out = CHARTS_DIR / 'chart2_per_topic.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out.name}')

# ── Chart 3: Similarity Score Distribution ────────────────────────────────────
def chart_similarity_distribution():
    ok_results  = [r for r in results if 'error' not in r or r.get('top_similarity', 0) > 0]
    sims        = [r['top_similarity'] for r in ok_results]
    hit_sims    = [r['top_similarity'] for r in ok_results if r['hit_at_k']]
    miss_sims   = [r['top_similarity'] for r in ok_results if not r['hit_at_k']]

    fig, ax = plt.subplots(figsize=(8, 4))

    bins = [i * 0.05 for i in range(0, 21)]   # 0.0 to 1.0 in steps of 0.05

    ax.hist(hit_sims,  bins=bins, alpha=0.75, color=SUCCESS, label='Hit (expected doc found)', zorder=3)
    ax.hist(miss_sims, bins=bins, alpha=0.75, color=FAIL,    label='Miss (expected doc not found)', zorder=3)
    ax.axvline(summary['threshold'], color=WARN, linestyle='--', linewidth=1.5,
               label=f'Threshold ({summary["threshold"]})', zorder=4)
    ax.axvline(sum(sims)/len(sims) if sims else 0, color=NEUTRAL, linestyle=':',
               linewidth=1.5, label=f'Mean ({sum(sims)/len(sims):.3f})', zorder=4)

    ax.set_xlabel('Top-1 Similarity Score', color=GRAY1)
    ax.set_ylabel('Number of Questions', color=GRAY1)
    ax.set_title('Distribution of Top-1 Cosine Similarity Scores', fontweight='bold', color=NEUTRAL)
    ax.legend(fontsize=8)
    ax.tick_params(colors=GRAY1)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    fig.tight_layout()

    out = CHARTS_DIR / 'chart3_similarity_distribution.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out.name}')

# ── Chart 4: Pass / Fail per Question ────────────────────────────────────────
def chart_question_heatmap():
    """Horizontal bar showing each question's similarity, coloured by hit/miss."""
    ok_results = [r for r in results if 'error' not in r]
    labels = [f"Q{r['id']:02d}: {r['question'][:42]}…" if len(r['question']) > 42
              else f"Q{r['id']:02d}: {r['question']}" for r in ok_results]
    sims   = [r['top_similarity'] for r in ok_results]
    colors = [SUCCESS if r['hit_at_k'] else FAIL for r in ok_results]

    fig, ax = plt.subplots(figsize=(10, max(6, len(ok_results) * 0.3)))
    y_pos   = range(len(ok_results))

    bars = ax.barh(list(y_pos), sims, color=colors, height=0.7, zorder=3)
    ax.axvline(summary['threshold'], color=WARN, linestyle='--', linewidth=1.2,
               label=f'Threshold ({summary["threshold"]})', zorder=4)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel('Top-1 Similarity Score', color=GRAY1)
    ax.set_title('Per-Question Retrieval Result', fontweight='bold', color=NEUTRAL)
    ax.invert_yaxis()

    hit_patch  = mpatches.Patch(color=SUCCESS, label='Hit (expected doc in top K)')
    miss_patch = mpatches.Patch(color=FAIL,    label='Miss')
    ax.legend(handles=[hit_patch, miss_patch], fontsize=8, loc='lower right')

    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.grid(axis='y', alpha=0)
    fig.tight_layout()

    out = CHARTS_DIR / 'chart4_per_question.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out.name}')

# ── Chart 5: Retrieved vs Failed Pie ─────────────────────────────────────────
def chart_retrieval_pie():
    retrieved = summary['retrieved_above_threshold']
    failed    = summary['failed_retrieval']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Pie 1: Retrieved vs failed
    ax1.pie(
        [retrieved, failed],
        labels=[f'Retrieved\n({retrieved})', f'Below threshold\n({failed})'],
        colors=[SUCCESS, GRAY2],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 9},
    )
    ax1.set_title(f'Retrieval Rate (threshold={summary["threshold"]})',
                  fontweight='bold', color=NEUTRAL)

    # Pie 2: Hit@K vs miss
    hit_k    = summary['hit_at_k']
    miss_k   = summary['successful_embed'] - hit_k
    ax2.pie(
        [hit_k, miss_k],
        labels=[f'Hit@{summary["top_k"]}\n({hit_k})', f'Miss\n({miss_k})'],
        colors=[ACCENT, GRAY2],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 9},
    )
    ax2.set_title(f'Hit@{summary["top_k"]} Rate\n(expected doc in top {summary["top_k"]})',
                  fontweight='bold', color=NEUTRAL)

    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    fig.suptitle('Retrieval Success Summary', fontsize=11, color=GRAY1)
    fig.tight_layout()

    out = CHARTS_DIR / 'chart5_retrieval_pie.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out.name}')

# ── Run All Charts ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\nGenerating charts from data/eval_results.json ...\n')
    chart_overall_summary()
    chart_per_topic_hitk()
    chart_similarity_distribution()
    chart_question_heatmap()
    chart_retrieval_pie()
    print('\nAll charts saved to -> data/eval_charts/')
    print('  chart1_overall_summary.png    -- headline metrics')
    print('  chart2_per_topic.png          -- hit rate + avg similarity per topic')
    print('  chart3_similarity_distribution.png -- similarity score histogram')
    print('  chart4_per_question.png       -- per-question pass/fail bar')
    print('  chart5_retrieval_pie.png      -- retrieval vs threshold pie')
