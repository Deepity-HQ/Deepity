import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# --- FONT UPDATE: Added Alliance No.2 to the front ---
plt.rcParams['font.sans-serif'] = ['Alliance No.2', 'Alliance No 2', 'Alliance', 'Arial', 'Helvetica', 'DejaVu Sans']
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

# --- Theme (Palantir-style Light Mode) ---
BG = "#F8FAFC"
TEXT = "#0F172A"
MUTED = "#64748B"
GRID = "#CBD5E1"
CURVE = "#0284C7"
GOLD = "#D97706"
GRAY = "#94A3B8"
RED = "#DC2626"
CARD_BG = "#FFFFFF"

fig = plt.figure(figsize=(16, 9), dpi=300, facecolor=BG)

# --- Enhanced 3D Studio Room (Infinity Cove) ---
bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
bg_ax.axis('off')

horizon = 0.15 # Aligns with the bottom of the graph axes

# 1. Cyclorama base
for i in range(200):
    y1, y2 = i/200.0, (i+1)/200.0
    if y1 <= horizon:
        progress = y1 / horizon
        c = 0.93 + 0.07 * progress 
    else:
        progress = (y1 - horizon) / (1.0 - horizon)
        c = 1.00 - 0.05 * progress 
    bg_ax.fill_between([0, 1], y1, y2, color=(c, c, c), zorder=1)

# 2. Seamless Full-Width Background Grid
# This replaces `ax.grid()` to fix the "vertical rectangle box" issue.
y_ticks_manual = [80, 85, 90, 95, 100]
for y_val in y_ticks_manual:
    # Map graph coordinates to figure coordinates (75 to 102 mapped into 0.15 to 0.75 height)
    y_fig = horizon + (y_val - 75) / (102 - 75) * 0.60
    # Draw line from entirely left to entirely right edge of image
    bg_ax.plot([0, 1], [y_fig, y_fig], color=GRID, alpha=0.4, linewidth=1, linestyle="--", zorder=2)

# 3. Floor/Wall spotlights
for i in range(40):
    radius = 0.6 * (40 - i) / 40
    alpha = 0.04 * (1 - i/40)
    bg_ax.add_patch(matplotlib.patches.Ellipse((0.5, horizon), radius*2, radius*0.4, 
                                               color='#FFFFFF', alpha=alpha, transform=bg_ax.transAxes, ec="none", zorder=3))
    bg_ax.add_patch(plt.Circle((0.5, 0.5), radius*1.2, color='#FFFFFF', alpha=alpha*0.8, 
                               transform=bg_ax.transAxes, ec="none", zorder=3))


# --- Main Plot ---
ax = fig.add_axes([0.08, 0.15, 0.84, 0.60])
ax.set_facecolor('none')

# --- 1 & 2. Background Reference Lines (Flat Transparent Indicators) ---
ax.axhline(pytorch_ffnn_test, color=GOLD, linewidth=2.5, linestyle=":", alpha=0.25, zorder=2)
ax.axhline(old_impl_test, color=RED, linewidth=2, linestyle="-.", alpha=0.25, zorder=2)

ax.plot(ngc_epochs, ngc_acc, color=GRAY, linewidth=3.5, linestyle="--", solid_capstyle="round", alpha=0.25, zorder=4)

# --- 3. Headline Curve (DKPPCN) Translucent Depth (Frosted Glass & Depth of Field) ---
dx = 1.0    # Deeper X stretch
dy = -1.5   # Deeper Y stretch
micro_steps = 250

for i in range(micro_steps):
    prop = i / micro_steps
    alpha = 0.05 * (1 - prop)**2
    lw = 3.5 + (prop * 8)
    shift_x = dkp_epochs + dx * prop
    shift_y = [val + dy * prop for val in dkp_acc]
    ax.plot(shift_x, shift_y, color=CURVE, linewidth=lw, linestyle="-", solid_capstyle="round", alpha=alpha, zorder=5)

ax.fill_between(dkp_epochs, dkp_acc, 75, color=CURVE, alpha=0.03, zorder=6)

ax.plot(dkp_epochs, dkp_acc, color=CURVE, linewidth=4.5, solid_capstyle="round", zorder=7)
ax.plot(dkp_epochs, dkp_acc, color='#BAE6FD', linewidth=1.2, solid_capstyle="round", zorder=8)
# Just a tiny touch of white for that glass edge reflection
ax.plot(dkp_epochs, [v + 0.1 for v in dkp_acc], color='#FFFFFF', linewidth=0.6, solid_capstyle="round", zorder=9)

# --- 4. End Marker (Flat Indicator) ---
ax.scatter(dkp_epochs[-1], dkp_acc[-1], s=180, color=CURVE, edgecolors=CARD_BG, linewidth=2.5, zorder=10)

# --- Direct line labels ---
# We keep labels fully opaque for readability
ax.text(3, 99, f'PyTorch Backprop: {pytorch_ffnn_test}%',
        color=GOLD, fontsize=13, va='bottom', ha='left', weight='bold')
ax.text(15.5, 91.8, f'Previous Implementation: {old_impl_test}% (~{old_time_min} min)',
        color=RED, fontsize=13, va='top', ha='left', weight='bold', alpha=0.9)
ax.text(15.5, 96, f'ngc-learn: {ngc_test}%',
        color=MUTED, fontsize=12.5, va='top', ha='left')

# --- Direct label for the headline curve ---
ax.text(48.5, 101.3, f'Deepity DKPPCN: {dkp_test}%  ({dkp_time}s)',
        color=CURVE, fontsize=15, weight="bold", ha='right', va='top', zorder=11)

# --- Axes styling ---
ax.set_xlim(0, 52)
ax.set_ylim(75, 102)
# Force our ticks to match the background seamless grid lines
ax.set_yticks(y_ticks_manual)

for spine in ax.spines.values():
    spine.set_visible(False)
# Removed inner ax.grid() to prevent the "vertical rectangle" artifact completely
ax.tick_params(colors=MUTED, labelsize=12, length=0, pad=10)
ax.set_xlabel("Training Epoch", color=TEXT, fontsize=14, labelpad=15, weight="bold")
ax.set_ylabel("MNIST Test Accuracy (%)", color=TEXT, fontsize=14, labelpad=15, weight="bold")

# --- Header ---
fig.text(0.08, 0.88, "An Alternative to Backprop Just Closed the Gap",
         color=TEXT, fontsize=34, weight="bold")
fig.text(0.08, 0.82, "Deepity DKPPCN • Predictive Coding Network Trained Locally in C++",
         color=CURVE, fontsize=16, weight="bold", alpha=0.95)

# --- Stats card ---
card_text = (
    f"{dkp_test}% Test Accuracy\n"
    f"{dkp_time}s Total Training Time\n"
    f"{speedup}x Faster than Previous Version\n\n"
    "✓ 60,000 training images\n"
    "✓ Trained locally in C++\n"
    "✓ No backpropagation required"
)

# Soft shadow for card
ax.text(
    0.903, 0.292, card_text,
    transform=ax.transAxes, color=(0,0,0,0), fontsize=13.5, linespacing=1.6,
    va="center", ha="right",
    bbox=dict(boxstyle="round,pad=1.1", facecolor='black', edgecolor='none', alpha=0.04), zorder=12,
)
# Top Card
ax.text(
    0.90, 0.30, card_text,
    transform=ax.transAxes, color=TEXT, fontsize=13.5, linespacing=1.6,
    va="center", ha="right",
    bbox=dict(boxstyle="round,pad=1.1", facecolor=CARD_BG, edgecolor=GRID, linewidth=1.5, alpha=0.95), zorder=13,
)

# --- Footer ---
fig.text(0.08, 0.05,
         f"DKPPCN closes to within {pytorch_ffnn_test - dkp_test:.2f} points of backprop while training {speedup}x faster than the previous implementation.",
         fontsize=13, color=MUTED)
fig.text(0.92, 0.05, "LinkedIn: @jack-c-rose  |  github.com/Ra4ster",
         fontsize=12, color=MUTED, alpha=0.7, ha="right")

plt.savefig("resources/dkppcn_chart.png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.4)
plt.close(fig)
print("Saved resources/dkppcn_chart.png")