# Load required libraries
library(tidyverse)
library(ggplot2)
library(patchwork)
library(scales)
library(cowplot)

# Set seed for reproducibility
set.seed(42)

# DATA LOADING AND PREPARATION
data <- read.csv("all_tables.csv", stringsAsFactors = FALSE)

# Create long format for all data
data_long <- data %>%
  pivot_longer(
    cols = c(Agent1_Delta, Agent2_Delta, Agent1_RPDI, Agent2_RPDI, 
             Agent1_Avg_Prices, Agent2_Avg_Prices),
    names_to = c("Agent", ".value"),
    names_pattern = "(Agent[12])_(.*)"
  ) %>%
  rename(Delta = Delta, RPDI = RPDI, Avg_Prices = Avg_Prices)

# Extract algorithm
data_long <- data_long %>%
  mutate(
    Algorithm = case_when(
      Agent == "Agent1" ~ str_extract(Matchup, "^[A-Z]+"),
      Agent == "Agent2" ~ str_extract(Matchup, "vs ([A-Z]+)$", group = 1)
    ),
    Algorithm = case_when(
      Algorithm == "Q" ~ "Q-learning",
      Algorithm == "DQN" ~ "DDQN",
      Algorithm == "PSO" ~ "PSO",
      Algorithm == "DDPG" ~ "DDPG",
      TRUE ~ Algorithm
    ),
    Algorithm = factor(Algorithm, levels = c("Q-learning", "DDQN", "PSO", "DDPG")),
    Shock_Condition = case_when(
      Shock == "0" ~ "No Shock",
      Shock == "A" ~ "Shock A",
      Shock == "B" ~ "Shock B",
      Shock == "C" ~ "Shock C"
    ),
    Shock_Condition = factor(Shock_Condition, 
                              levels = c("No Shock", "Shock A", "Shock B", "Shock C")),
    Model = factor(Model, levels = c("LOGIT", "HOTELLING", "LINEAR"))
  )

# PREPARE PRICE DATA BY MODEL
price_data_all <- data_long %>%
  group_by(Model, Algorithm, Shock_Condition) %>%
  summarise(
    Mean_Price = mean(Avg_Prices, na.rm = TRUE),
    SE_Price = sd(Avg_Prices, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

# Theoretical Nash prices for each model
theo_nash_logit <- data.frame(
  Shock_Condition = factor(c("No Shock", "Shock A", "Shock B", "Shock C"),
                           levels = c("No Shock", "Shock A", "Shock B", "Shock C")),
  Nash_Price = c(1.47, 1.80, 1.54, 1.91)
)

theo_nash_hotelling <- data.frame(
  Shock_Condition = factor(c("No Shock", "Shock A", "Shock B", "Shock C"),
                           levels = c("No Shock", "Shock A", "Shock B", "Shock C")),
  Nash_Price = c(1.0, 1.0, 1.0, 1.0)  # Hotelling Nash is constant
)

theo_nash_linear <- data.frame(
  Shock_Condition = factor(c("No Shock", "Shock A", "Shock B", "Shock C"),
                           levels = c("No Shock", "Shock A", "Shock B", "Shock C")),
  Nash_Price = c(0.43, 0.43, 0.43, 0.43)  # Linear Nash is constant
)

# COLOR PALETTE (Colorblind-friendly)
algo_colors <- c(
  "Q-learning" = "#1B9E77",
  "DDQN"       = "#D95F02",
  "PSO"        = "#7570B3", 
  "DDPG"       = "#E7298A"
)

algo_shapes <- c(
  "Q-learning" = 16,
  "DDQN"        = 15,
  "PSO"        = 17,
  "DDPG"       = 18
)

algo_linetypes <- c(
  "Q-learning" = "solid",
  "DDQN"       = "solid",
  "PSO"        = "solid",
  "DDPG"       = "solid"
)

# THEME (Cross-platform compatible)
theme_q1_journal <- function(base_size = 10) {
  theme_bw(base_size = base_size) +
    theme(
      text = element_text(family = 'serif'),
      plot.title = element_text(
        size = rel(1.15), 
        face = "bold", 
        hjust = 0.5,
        margin = margin(b = 4)
      ),
      plot.subtitle = element_text(
        size = rel(0.85),
        hjust = 0.5,
        margin = margin(b = 6),
        color = "gray40",
        face = "italic"
      ),
      axis.title = element_text(size = rel(1.0), face = "plain"),
      axis.title.x = element_text(margin = margin(t = 8)),
      axis.title.y = element_text(margin = margin(r = 8)),
      axis.text = element_text(size = rel(0.9), color = "black"),
      panel.grid.major = element_line(color = "gray88", linewidth = 0.35),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.6),
      legend.title = element_text(size = rel(0.95), face = "bold"),
      legend.text = element_text(size = rel(0.9)),
      legend.key.size = unit(1.0, "lines"),
      legend.key.width = unit(1.5, "lines"),
      legend.background = element_rect(fill = "white", color = NA),
      legend.key = element_rect(fill = "white", color = NA),
      plot.margin = margin(t = 8, r = 8, b = 8, l = 8)
    )
}

# PANEL 1: LOGIT - Price vs Benchmark
price_logit <- price_data_all %>% filter(Model == "LOGIT")

p1_logit <- ggplot() +
  # Reference: No-Shock Nash horizontal line
  geom_hline(yintercept = 1.47, linetype = "dotted", 
             color = "gray55", linewidth = 0.5) +
  # Theoretical Nash prices (Black dashed line with X markers)
  geom_line(data = theo_nash_logit, 
            aes(x = Shock_Condition, y = Nash_Price, group = 1),
            linetype = "longdash", linewidth = 1.1, color = "black") +
  geom_point(data = theo_nash_logit,
             aes(x = Shock_Condition, y = Nash_Price),
             size = 3, shape = 4, stroke = 1.5, color = "black") +
  # Algorithm actual prices
  geom_line(data = price_logit, 
            aes(x = Shock_Condition, y = Mean_Price, 
                color = Algorithm, group = Algorithm),
            linewidth = 0.9) +
  geom_point(data = price_logit,
             aes(x = Shock_Condition, y = Mean_Price, 
                 color = Algorithm, shape = Algorithm),
             size = 2.5, stroke = 0.4) +
  # Scales
  scale_color_manual(values = algo_colors) +
  scale_shape_manual(values = algo_shapes) +
  scale_y_continuous(
    limits = c(1.35, 2.00),
    breaks = seq(1.4, 2.0, by = 0.2),
    minor_breaks = NULL
  ) + labs(y = "Nash Price", x = element_blank()) +
  # Annotations
  annotate("text", x = 2.5, y = 1.78, 
           label = "Theo. Nash", 
           size = 2.5, color = "black", angle = -60) +
  annotate("text", x = 4.4, y = 1.44, 
           label = "No-shock", 
           size = 2.2, color = "gray55", hjust = 1) +
  theme_q1_journal() +
  theme(
    legend.position = "none",
    plot.subtitle = element_text(color = "#D55E00"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8)
  )

# PANEL 2: HOTELLING - Price vs Benchmark
price_hotelling <- price_data_all %>% filter(Model == "HOTELLING")

p2_hotelling <- ggplot() +
  # Theoretical Nash prices (constant at 1.0)
  geom_hline(yintercept = 1.0, linetype = "longdash",
             color = "black", linewidth = 1.1) +
  geom_point(data = theo_nash_hotelling,
             aes(x = Shock_Condition, y = Nash_Price),
             size = 3, shape = 4, stroke = 1.5, color = "black") +
  # Algorithm actual prices
  geom_line(data = price_hotelling,
            aes(x = Shock_Condition, y = Mean_Price,
                color = Algorithm, group = Algorithm),
            linewidth = 0.9) +
  geom_point(data = price_hotelling,
             aes(x = Shock_Condition, y = Mean_Price,
                 color = Algorithm, shape = Algorithm),
             size = 2.5, stroke = 0.4) +
  # Scales
  scale_color_manual(values = algo_colors) +
  scale_shape_manual(values = algo_shapes) +
  scale_y_continuous(
    limits = c(0.95, 1.25),
    breaks = seq(0.95, 1.25, by = 0.1),
    minor_breaks = NULL
  ) + labs(y = "Nash Price", x = element_blank()) +
  # Annotations
  annotate("text", x = 1.0, y = 0.97, 
           label = "Theo. Nash", 
           size = 2.5, color = "black", hjust = 0) +
  theme_q1_journal() +
  theme(
    legend.position = "none",
    plot.subtitle = element_text(color = "#009E73"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8)
  )

# PANEL 3: LINEAR - Price vs Benchmark
price_linear <- price_data_all %>% filter(Model == "LINEAR")

p3_linear <- ggplot() +
  # Theoretical Nash prices (constant at 0.43)
  geom_hline(yintercept = 0.43, linetype = "longdash", 
             color = "black", linewidth = 1.1) +
  geom_point(data = theo_nash_linear,
             aes(x = Shock_Condition, y = Nash_Price),
             size = 3, shape = 4, stroke = 1.5, color = "black") +
  # Algorithm actual prices
  geom_line(data = price_linear, 
            aes(x = Shock_Condition, y = Mean_Price, 
                color = Algorithm, group = Algorithm),
            linewidth = 0.9) +
  geom_point(data = price_linear,
             aes(x = Shock_Condition, y = Mean_Price, 
                 color = Algorithm, shape = Algorithm),
             size = 2.5, stroke = 0.4) +
  # Scales
  scale_color_manual(values = algo_colors) +
  scale_shape_manual(values = algo_shapes) +
  scale_y_continuous(
    limits = c(0.42, 0.52),
    breaks = seq(0.42, 0.52, by = 0.02),
    minor_breaks = NULL
  ) + labs(y = "Nash Price", x = element_blank()) +
  # Annotations
  annotate("text", x = 1.0, y = 0.4225, 
           label = "Theo. Nash", 
           size = 2.5, color = "black", hjust = 0) +
  theme_q1_journal() +
  theme(
    legend.position = "none",
    plot.subtitle = element_text(color = "#CC79A7"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8)
  )

# CREATE SHARED LEGEND
p_legend <- ggplot(price_data_all, aes(x = Shock_Condition, y = Mean_Price)) +
  geom_line(aes(color = Algorithm, group = Algorithm), linewidth = 0.9) +
  geom_point(aes(color = Algorithm, shape = Algorithm), size = 2.5) +
  scale_color_manual(values = algo_colors, name = "Algorithm") +
  scale_shape_manual(values = algo_shapes, name = "Algorithm") +
  guides(
    color = guide_legend(title.position = "top", ncol = 1),
    shape = guide_legend(title.position = "top", ncol = 1)
  ) +
  theme_q1_journal() +
  theme(
    legend.title = element_text(size = 9, face = "bold"),
    legend.text = element_text(size = 8)
  )

legend_grob <- get_legend(p_legend)

# COMBINE ALL PANELS
# Add common x-axis label
x_label <- ggdraw() +
  draw_label("Shock Condition", size = 10, fontfamily = "serif")

# Combine the three plots
combined_plots <- (p1_logit | p2_hotelling | p3_linear) +
  plot_layout(ncol = 3, widths = c(1, 1, 1))

# Add x-axis label below plots
plots_with_xlabel <- plot_grid(
  combined_plots,
  x_label,
  ncol = 1,
  rel_heights = c(1, 0.05)
)

# Final assembly with legend on the right
final_figure <- plot_grid(
  plots_with_xlabel,
  legend_grob,
  ncol = 2,
  rel_widths = c(1, 0.12),
  align = "h",
  axis = "tb"
)

# Add overall title
title_grob <- ggdraw() + 
  draw_label(
    "Price Stability vs Theoretical Benchmark Across Market Structures",
    fontface = 'bold',
    size = 12,
    fontfamily = 'serif'
  )

# Final plot with title
final_plot <- plot_grid(
  title_grob,
  final_figure,
  ncol = 1,
  rel_heights = c(0.06, 1)
) + theme(plot.margin = margin(5, 10, 5, 5))

ggsave(
  filename = "./figures/Figure3_Price_vs_Benchmark_AllMarkets.png",
  plot = final_plot,
  width = 180,
  height = 75,
  units = "mm",
  dpi = 600,
  bg = "white"
)

ggsave(
  filename = "./figures/Figure3_Price_vs_Benchmark_AllMarkets.pdf",
  plot = final_plot,
  width = 180,
  height = 75,
  units = "mm",
  device = "pdf"
)

cat("\n")
cat("============================================================\n")
cat("  FIGURE 3: Price vs Benchmark - All Market Structures\n")
cat("============================================================\n")
cat("\n")
cat("Files created:\n")
cat("  - Figure3_Price_vs_Benchmark_AllMarkets.png  (600 DPI)\n")
cat("  - Figure3_Price_vs_Benchmark_AllMarkets.pdf  (Vector)\n")
cat("\n")
cat("Key patterns visualized:\n")
cat("  - Logit: Nash shifts (1.47->1.91), prices lag behind (~1.65)\n")
cat("  - Hotelling: Nash constant (1.0), prices stable above (~1.10)\n")
cat("  - Linear: Nash constant (0.43), prices slightly above (~0.45)\n")
cat("\n")
