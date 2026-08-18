import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "result1formatted.csv"
OUTPUT = "mnist_predictive_coding_linkedin.png"

BG = "#0F172A"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
GRID = "#334155"
CURVE = "#06B6D4"
CARD_BG = "#1E293B"

plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'

df = pd.read_csv(CSV_FILE)

epochs = df["epoch"]
energy = df["energy"]
time_taken = df["time_taken"]

total_time_mins = time_taken.sum() / 60
avg_time_per_epoch = time_taken.mean()

fig = plt.figure(figsize=(16, 9), dpi=300, facecolor=BG)
ax = fig.add_axes([0.08, 0.15, 0.84, 0.60])
ax.set_facecolor(BG)

for lw, alpha in [(20, 0.03), (14, 0.05), (10, 0.08), (7, 0.13)]:
    ax.plot(
        epochs, energy,
        color=CURVE, linewidth=lw, alpha=alpha, solid_capstyle="round"
    )

ax.plot(epochs, energy, color=CURVE, linewidth=3.5, solid_capstyle="round")

ax.fill_between(epochs, energy, energy.min(), color=CURVE, alpha=0.08)

ax.scatter(
    epochs.iloc[-1], energy.iloc[-1],
    s=150, color=CURVE, edgecolors=TEXT, linewidth=2, zorder=10
)

for spine in ax.spines.values():
    spine.set_visible(False)

ax.grid(True, color=GRID, alpha=0.4, linewidth=1, linestyle="--")
ax.tick_params(colors=MUTED, labelsize=12, length=0, pad=10)

ax.set_xlabel("Training Epoch", color=MUTED, fontsize=14, labelpad=15, weight="bold")
ax.set_ylabel("Batch Energy", color=MUTED, fontsize=14, labelpad=15, weight="bold")

fig.text(
    0.08, 0.88, "Energy Minimization Can Learn MNIST",
    color=TEXT, fontsize=34, weight="bold"
)
fig.text(
    0.08, 0.82, "Predictive Coding Network • Energy Convergence During Training",
    color=CURVE, fontsize=16, weight="bold", alpha=0.9
)

card_text = (
    "92.42% Test Accuracy\n"
    f"{total_time_mins:.1f}m Total Training Time\n"
    f"{avg_time_per_epoch:.1f}s Average per Epoch\n\n"
    "✓ 70,000 training images\n"
    "✓ Stable energy convergence\n"
    "✓ No backpropagation required"
)

ax.text(
    0.97, 0.94, card_text,
    transform=ax.transAxes,
    color=TEXT, fontsize=14, linespacing=1.6,
    va="top", ha="right",
    bbox=dict(
        boxstyle="round,pad=1.2", facecolor=CARD_BG,
        edgecolor=GRID, linewidth=1.5, alpha=0.95
    )
)

ax.annotate(
    f"Converged at e={energy.iloc[-1]:.0f}",
    xy=(epochs.iloc[-1], energy.iloc[-1]),
    xytext=(-100, 45), textcoords="offset points",
    fontsize=14, color=TEXT, weight="bold",
    arrowprops=dict(arrowstyle="->", color=CURVE, lw=2, connectionstyle="arc3,rad=0.2")
)

fig.text(
    0.08, 0.05,
    "Batch energy decreases consistently throughout training, indicating stable optimization.",
    fontsize=13, color=MUTED
)

fig.text(
    0.92, 0.05, "LinkedIn: @jack-c-rose | github.com/Ra4ster",
    fontsize=12, color=MUTED, alpha=0.5, ha="right"
)

plt.savefig(
    OUTPUT, facecolor=fig.get_facecolor(),
    bbox_inches="tight", pad_inches=0.4
)
plt.close(fig)