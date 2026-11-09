#!/usr/bin/env Rscript
# ggplot2 remake of plot_linkedin_graph.py -- same data, same dark social-card
# look, but real vector geoms (geom_line/geom_area) instead of hand-rolled
# glow layers, and ggplot2's own coordinate system for the annotations.

library(ggplot2)
library(scales)
library(grid)

# Absolute, not relative to cwd/script location: cwd varies by how this
# gets run (Rscript, source(), or pasted line-by-line into a console), and
# only Rscript's own invocation exposes a --file= arg to derive a path from.
REPO_DIR <- "/home/rose0/Projects/deepity/experiments"
CSV_FILE <- file.path(REPO_DIR, "outputs/result1formatted.csv")
OUTPUT   <- file.path(REPO_DIR, "outputs/mnist_predictive_coding_linkedin_ggplot.png")

BG      <- "#0F172A"
TEXT    <- "#F8FAFC"
MUTED   <- "#94A3B8"
GRID_CL <- "#334155"
CURVE   <- "#06B6D4"
CARD_BG <- "#1E293B"

energy_df <- read.csv(CSV_FILE)

total_time_mins     <- sum(energy_df$time_taken) / 60
avg_time_per_epoch  <- mean(energy_df$time_taken)
final_epoch         <- energy_df$epoch[nrow(energy_df)]
final_energy        <- energy_df$energy[nrow(energy_df)]

card_text <- paste0(
  "92.42% Test Accuracy\n",
  sprintf("%.1f", total_time_mins), "m Total Training Time\n",
  sprintf("%.1f", avg_time_per_epoch), "s Average per Epoch\n\n",
  "✓ 70,000 training images\n",
  "✓ Stable energy convergence\n",
  "✓ No backpropagation required"
)

p <- ggplot(energy_df, aes(epoch, energy)) +
  # Glow: the same curve redrawn wide/faint underneath, like the matplotlib
  # version's repeated ax.plot() calls -- geom_line() just takes a vector of
  # linewidths/alphas instead of four separate calls.
  lapply(list(c(20, 0.03), c(14, 0.05), c(10, 0.08), c(7, 0.13)), function(s) {
    geom_line(colour = CURVE, linewidth = s[1], alpha = s[2], lineend = "round")
  }) +
  geom_area(fill = CURVE, alpha = 0.08) +
  geom_line(colour = CURVE, linewidth = 1.1, lineend = "round") +
  geom_point(
    data = energy_df[nrow(energy_df), ], aes(epoch, energy),
    colour = CURVE, fill = CURVE, shape = 21, size = 4.5,
    stroke = 1.3
  ) +
  annotate(
    "label", x = final_epoch - 11, y = final_energy + max(energy_df$energy) * 0.07,
    label = sprintf("Converged at e=%.0f", final_energy),
    colour = TEXT, fill = BG, linewidth = 0, size = 4.6, fontface = "bold"
  ) +
  annotate(
    "curve", x = final_epoch - 7, y = final_energy + max(energy_df$energy) * 0.04,
    xend = final_epoch - 1, yend = final_energy + max(energy_df$energy) * 0.01,
    curvature = -0.2, colour = CURVE, linewidth = 0.7,
    arrow = arrow(length = unit(2.2, "mm"), type = "closed")
  ) +
  annotate(
    "label", x = max(energy_df$epoch), y = max(energy_df$energy) * 1.02,
    label = card_text, colour = TEXT, fill = CARD_BG,
    linewidth = 0.4, label.r = unit(3, "mm"), label.padding = unit(0.6, "lines"),
    hjust = 1, vjust = 1, size = 3.9, lineheight = 1.35
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.01, 0.01))) +
  scale_y_continuous(labels = label_number(big.mark = ","), expand = expansion(mult = c(0.02, 0.12))) +
  labs(
    title    = "Energy Minimization Can Learn MNIST",
    subtitle = "Predictive Coding Network • Energy Convergence During Training",
    x = "Training Epoch", y = "Batch Energy",
    caption = "Batch energy decreases consistently throughout training, indicating stable optimization.\nLinkedIn: @jack-c-rose | github.com/Ra4ster"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.background  = element_rect(fill = BG, colour = NA),
    panel.background = element_rect(fill = BG, colour = NA),
    panel.grid.major = element_line(colour = GRID_CL, linewidth = 0.4, linetype = "dashed"),
    panel.grid.minor = element_blank(),
    axis.text  = element_text(colour = MUTED, size = 11),
    axis.title = element_text(colour = MUTED, size = 13, face = "bold"),
    axis.title.x = element_text(margin = margin(t = 12)),
    axis.title.y = element_text(margin = margin(r = 12)),
    axis.ticks = element_blank(),
    plot.title = element_text(colour = TEXT, size = 26, face = "bold", margin = margin(b = 4)),
    plot.subtitle = element_text(colour = CURVE, size = 14, face = "bold", margin = margin(b = 16)),
    plot.caption = element_text(colour = MUTED, size = 10, hjust = 0, lineheight = 1.4, margin = margin(t = 14)),
    plot.margin = margin(t = 20, r = 24, b = 10, l = 20)
  )

# Plain ggsave rather than a hand-opened png()/grid.draw() device: the latter
# depends on whatever graphics backend R picked as default on a given
# machine (Cairo, Xlib, ragg, ...) and isn't guaranteed to render a full
# ggplotGrob correctly on all of them -- ggsave() is the tested path.
ggsave(OUTPUT, p, width = 16, height = 9, dpi = 300, bg = BG)
