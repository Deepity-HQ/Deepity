import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'

# --- Data ---
dkp_epochs = np.arange(1, 51)
dkp_acc = [89.36, 93.44, 94.16, 94.48, 95.44, 96.52, 96.40, 96.56, 96.84, 97.52,
           97.68, 97.32, 97.72, 97.76, 97.96, 97.80, 98.44, 98.32, 97.80, 98.80,
           98.80, 98.44, 98.64, 98.68, 98.32, 98.32, 98.64, 99.04, 98.80, 99.16,
           98.96, 99.24, 99.28, 99.12, 98.96, 99.20, 99.40, 99.04, 99.32, 99.36,
           98.72, 99.32, 98.96, 99.24, 99.28, 99.36, 99.20, 99.12, 99.36, 99.48]
dkp_test = 97.73
dkp_time = 60

ngc_epochs = np.arange(1, 16)
ngc_acc = [26.91, 42.96, 60.12, 75.20, 84.68, 89.52, 91.90,
           93.45, 94.30, 94.80, 95.13, 95.38, 95.63, 95.74, 95.95]
ngc_test = 95.09

pytorch_ffnn_test = 98.27
old_impl_test = 92.42
old_time_min = 50
speedup = round(old_time_min * 60 / dkp_time)

# --- Theme (matched to the other chart) ---
BG = "#0F172A"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
GRID = "#334155"
CURVE = "#22D3EE"
GOLD = "#FBBF24"
GRAY = "#64748B"
RED = "#F87171"
CARD_BG = "#1E293B"

fig = plt.figure(figsize=(16, 9), dpi=300, facecolor=BG)
ax = fig.add_axes([0.08, 0.15, 0.84, 0.60])
ax.set_facecolor(BG)

# --- Glow layers for the headline curve ---
for lw, alpha in [(20, 0.03), (14, 0.05), (10, 0.08), (7, 0.13)]:
    ax.plot(dkp_epochs, dkp_acc, color=CURVE, linewidth=lw, alpha=alpha, solid_capstyle="round")
ax.plot(dkp_epochs, dkp_acc, color=CURVE, linewidth=3.5, solid_capstyle="round", zorder=6)
ax.fill_between(dkp_epochs, dkp_acc, 75, color=CURVE, alpha=0.08, zorder=1)

# --- Reference curves / lines ---
ax.plot(ngc_epochs, ngc_acc, linewidth=2.5, linestyle="--", color=GRAY, zorder=3, solid_capstyle="round")
ax.axhline(pytorch_ffnn_test, color=GOLD, linewidth=2, linestyle=":", zorder=2)
ax.axhline(old_impl_test, color=RED, linewidth=1.8, linestyle="-.", zorder=2, alpha=0.85)

ax.scatter(dkp_epochs[-1], dkp_acc[-1], s=150, color=CURVE, edgecolors=TEXT, linewidth=2, zorder=10)

# --- Direct line labels ---
ax.text(3, 100, f'PyTorch Backprop: {pytorch_ffnn_test}%',
        color=GOLD, fontsize=13, va='top', ha='left', weight='bold')
ax.text(15.5, 91.8, f'Previous Implementation: {old_impl_test}% (~{old_time_min} min)',
        color=RED, fontsize=13, va='top', ha='left', weight='bold', alpha=0.9)
ax.text(15.5, 96, f'ngc-learn: {ngc_test}%',
        color=MUTED, fontsize=12.5, va='top', ha='left')

# --- Direct label for the headline curve (no arrow — avoids collision with the card) ---
ax.text(48.5, 101.3, f'Deepity DKPPCN: {dkp_test}%  ({dkp_time}s)',
        color=CURVE, fontsize=15, weight="bold", ha='right', va='top', zorder=8)

# --- Axes styling ---
ax.set_xlim(0, 52)
ax.set_ylim(75, 102)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.grid(True, color=GRID, alpha=0.4, linewidth=1, linestyle="--")
ax.tick_params(colors=MUTED, labelsize=12, length=0, pad=10)
ax.set_xlabel("Training Epoch", color=MUTED, fontsize=14, labelpad=15, weight="bold")
ax.set_ylabel("MNIST Test Accuracy (%)", color=MUTED, fontsize=14, labelpad=15, weight="bold")

# --- Header ---
fig.text(0.08, 0.88, "An Alternative to Backprop Just Closed the Gap",
          color=TEXT, fontsize=34, weight="bold")
fig.text(0.08, 0.82, "Deepity DKPPCN • Predictive Coding Network Trained Locally in C++",
          color=CURVE, fontsize=16, weight="bold", alpha=0.9)

# --- Stats card ---
card_text = (
    f"{dkp_test}% Test Accuracy\n"
    f"{dkp_time}s Total Training Time\n"
    f"{speedup}x Faster than Previous Version\n\n"
    "✓ 60,000 training images\n"
    "✓ Trained locally in C++\n"
    "✓ No backpropagation required"
)
ax.text(
    0.90, 0.30, card_text,
    transform=ax.transAxes,
    color=TEXT, fontsize=13.5, linespacing=1.6,
    va="center", ha="right",
    bbox=dict(boxstyle="round,pad=1.1", facecolor=CARD_BG, edgecolor=GRID, linewidth=1.5, alpha=0.95),
    zorder=11,
)

# --- Footer ---
fig.text(0.08, 0.03,
          f"DKPPCN closes to within {pytorch_ffnn_test - dkp_test:.2f} points of backprop while training {speedup}x faster than the previous implementation.",
          fontsize=13, color=MUTED)
fig.text(0.92, 0.09, "LinkedIn: @jack-c-rose  |  github.com/Ra4ster",
          fontsize=12, color=MUTED, alpha=0.5, ha="right")

plt.savefig("resources/dkppcn_chart.png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.4)
plt.close(fig)
print("Saved resources/dkppcn_chart.png")
