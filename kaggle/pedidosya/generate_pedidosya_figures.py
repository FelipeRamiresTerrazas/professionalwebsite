"""
Generate dark-themed portfolio figures for the PedidosYa friction analysis.
Reads: kaggle/pedidosya/pedidosya_friction_metrics.json
Saves: figures/pedidosya_*.png
"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE   = pathlib.Path(__file__).parent
FIG    = pathlib.Path(__file__).parent.parent.parent / "figures"
FIG.mkdir(exist_ok=True)
METRICS = json.loads((BASE / "pedidosya_friction_metrics.json").read_text(encoding="utf-8"))

plt.style.use("dark_background")
RED     = "#FF043C"
RED2    = "#CC0030"
ACCENT  = "#FF6B8A"
GRAY    = "#1E293B"
GRAY2   = "#334155"
LIGHT   = "#94A3B8"
WHITE   = "#E2E8F0"
BG      = "#0F172A"
GREEN   = "#22C55E"
AMBER   = "#F59E0B"

EXPORT_KW = dict(dpi=160, bbox_inches="tight",
                 facecolor=BG, edgecolor="none")

# ── 1. Score Distribution ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
ax.set_facecolor(BG)

sd = METRICS["meta"]["score_distribution"]
stars  = [1, 2, 3, 4, 5]
counts = [sd[str(s)] for s in stars]
total  = sum(counts)
colors = [RED, RED2, AMBER, "#4ADE80", GREEN]

bars = ax.barh(stars, counts, color=colors, height=0.55, zorder=3)
for bar, cnt, s in zip(bars, counts, stars):
    pct = 100 * cnt / total
    ax.text(cnt + 12, s, f"{cnt:,}  ({pct:.1f}%)",
            va="center", ha="left", color=LIGHT, fontsize=9.5)

ax.set_yticks(stars)
ax.set_yticklabels([f"{'★'*s}" for s in stars], color=WHITE, fontsize=11)
ax.set_xlabel("Number of reviews", color=LIGHT, labelpad=10)
ax.set_title("Rating Distribution — PedidosYa (BR, 2,000 reviews)",
             color=WHITE, fontsize=13, fontweight="bold", pad=14)
ax.set_xlim(0, max(counts) * 1.3)
ax.tick_params(axis="x", colors=LIGHT)
for spine in ax.spines.values(): spine.set_visible(False)
ax.xaxis.grid(True, color=GRAY2, linewidth=0.5, zorder=0)

# annotation box
ax.annotate("43.9% negative\n(1-2★)",
            xy=(sd["1"] + sd["2"], 1.5),
            xytext=(sd["1"] * 0.5, 3.2),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
            color=RED, fontsize=9.5, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc=GRAY, ec=RED, lw=1))

fig.tight_layout()
out = FIG / "pedidosya_score_distribution.png"
fig.savefig(out, **EXPORT_KW)
plt.close(fig)
print(f"[fig] Saved {out.name}")

# ── 2. Keyword Friction Frequency ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
ax.set_facecolor(BG)

kf = METRICS["keyword_freq"]
sorted_kf = sorted(kf.items(), key=lambda x: -x[1])
labels  = [k for k, v in sorted_kf]
values  = [v for k, v in sorted_kf]
neg_total = METRICS["meta"]["negative_reviews"]
bar_colors = [RED if v == max(values) else RED2 if v > 5 else GRAY2 for v in values]

bars = ax.barh(labels, values, color=bar_colors, height=0.55, zorder=3)
for bar, val, lbl in zip(bars, values, labels):
    pct = 100 * val / neg_total
    txt = f"{val} mentions  ({pct:.1f}% of neg. reviews)" if val > 0 else "0 mentions"
    ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
            txt, va="center", ha="left", color=LIGHT, fontsize=9)

ax.set_xlabel("Mentions in 1-2★ reviews", color=LIGHT, labelpad=10)
ax.set_title("Friction Keyword Frequency — Negative Reviews Only",
             color=WHITE, fontsize=13, fontweight="bold", pad=14)
ax.set_xlim(0, max(values) * 1.55 if max(values) > 0 else 10)
ax.tick_params(axis="both", colors=LIGHT)
for spine in ax.spines.values(): spine.set_visible(False)
ax.xaxis.grid(True, color=GRAY2, linewidth=0.5, zorder=0)
ax.invert_yaxis()

fig.tight_layout()
out = FIG / "pedidosya_keyword_friction.png"
fig.savefig(out, **EXPORT_KW)
plt.close(fig)
print(f"[fig] Saved {out.name}")

# ── 3. Top Organic Terms word-bar ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
ax.set_facecolor(BG)

tw = METRICS["top_words"][:15]
words  = [t["word"] for t in tw]
cnts   = [t["count"] for t in tw]
max_c  = max(cnts)
bar_cols = [RED if c == max_c else
            "#FF6B8A" if c > 60 else
            "#994466" if c > 40 else
            GRAY2 for c in cnts]

bars = ax.bar(words, cnts, color=bar_cols, zorder=3, width=0.65)
for bar, c in zip(bars, cnts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            str(c), ha="center", va="bottom", color=LIGHT, fontsize=8.5)

ax.set_title("Top 15 Organic Terms — Negative Reviews (1-2★)",
             color=WHITE, fontsize=13, fontweight="bold", pad=14)
ax.set_ylabel("Frequency", color=LIGHT, labelpad=8)
ax.tick_params(axis="x", rotation=30, colors=LIGHT)
ax.tick_params(axis="y", colors=LIGHT)
ax.set_ylim(0, max_c * 1.20)
for spine in ax.spines.values(): spine.set_visible(False)
ax.yaxis.grid(True, color=GRAY2, linewidth=0.5, zorder=0)

# highlight competitive signal
ifood_idx = words.index("ifood") if "ifood" in words else -1
if ifood_idx >= 0:
    ax.annotate("Competitor\nchurn signal",
                xy=(ifood_idx, cnts[ifood_idx]),
                xytext=(ifood_idx + 1.8, cnts[ifood_idx] + 18),
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.4),
                color=AMBER, fontsize=8.5, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc=GRAY, ec=AMBER, lw=1))

fig.tight_layout()
out = FIG / "pedidosya_top_terms.png"
fig.savefig(out, **EXPORT_KW)
plt.close(fig)
print(f"[fig] Saved {out.name}")

# ── 4. Funnel — symmetric horizontal bars that narrow like a real funnel ─────
stages = [
    ("App Open",          2000, "#22C55E", "100% — All users"),
    ("App Stable",        1677, "#4ADE80", "83.9% — Crash-free session"),
    ("Address Resolved",  1533, "#F59E0B", "76.7% — GPS resolves correctly"),
    ("Checkout Reached",  1380, "#FB923C", "69.0% — Reaches checkout"),
    ("Payment Success",   1198, "#FF043C", "59.9% — Order completes"),
]

fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=BG)
ax.set_facecolor(BG)

labels = [s[0] for s in stages]
values = [s[1] for s in stages]
colors = [s[2] for s in stages]
notes  = [s[3] for s in stages]
max_v  = values[0]

# Bar height and vertical spacing
bar_h   = 0.55
spacing = 1.0          # row centres at y = 0, 1, 2 ... (inverted later)
n       = len(stages)

for i, (lbl, val, col, note) in enumerate(stages):
    y      = (n - 1 - i) * spacing   # invert so "App Open" is at top
    half_w = val / max_v              # normalised half-width → symmetric funnel
    # Filled bar — symmetric around x=0
    ax.barh(y, val, left=0, height=bar_h, color=col, alpha=0.85, zorder=3)
    # Mirror bar (left side) — creates symmetric funnel shape
    ax.barh(y, -val, left=0, height=bar_h, color=col, alpha=0.85, zorder=3)

    # Stage label (left of bar)
    ax.text(-max_v * 1.03, y, lbl,
            va="center", ha="right", color=WHITE,
            fontsize=9.5, fontweight="bold")

    # Value + pct (inside bar, right side)
    ax.text(val * 0.5, y, f"{val:,}",
            va="center", ha="center", color="white",
            fontsize=9, fontweight="black")

    # Note (right of bar)
    ax.text(max_v * 1.03, y, note,
            va="center", ha="left", color=LIGHT, fontsize=8.5)

    # Drop-off arrow between stages
    if i < n - 1:
        next_val = values[i + 1]
        drop_pct = round(100 * (val - next_val) / val, 1)
        y_mid    = y - spacing / 2
        ax.annotate("", xy=(next_val, y - spacing + bar_h / 2 + 0.05),
                    xytext=(val, y - bar_h / 2 - 0.05),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY2,
                                   lw=1.3, mutation_scale=10))
        ax.annotate("", xy=(-next_val, y - spacing + bar_h / 2 + 0.05),
                    xytext=(-val, y - bar_h / 2 - 0.05),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY2,
                                   lw=1.3, mutation_scale=10))
        ax.text(0, y_mid, f"−{drop_pct}%",
                va="center", ha="center", color=RED,
                fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec=RED, lw=0.8))

ax.set_xlim(-max_v * 1.7, max_v * 1.95)
ax.set_ylim(-0.6, (n - 1) * spacing + 0.6)
ax.axis("off")
ax.set_title("Estimated User Funnel — Where Friction Kills Conversion",
             color=WHITE, fontsize=13, fontweight="bold", pad=14)
ax.text(0, -0.5,
        "Estimates derived from friction keyword rates + industry benchmarks.",
        ha="center", va="center", color=LIGHT, fontsize=8, style="italic")

fig.tight_layout()
out = FIG / "pedidosya_funnel.png"
fig.savefig(out, **EXPORT_KW)
plt.close(fig)
print(f"[fig] Saved {out.name}")

print("\n[generate] All 4 figures saved to figures/")
