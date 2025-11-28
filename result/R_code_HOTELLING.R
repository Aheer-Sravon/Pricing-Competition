# =============================================================================
# HOTELLING MODEL: Comprehensive Three-Panel Figure for Q1 Journal Publication
# =============================================================================
# This script creates a publication-quality figure with three panels:
# 1. Delta vs RPDI Scatter Plot
# 2. Shock Impact Across Conditions (Line Plot)
# 3. Price Stability vs Benchmark Shift
# With a shared legend on the right side
# =============================================================================

# Load required libraries
library(tidyverse)
library(ggplot2)
library(patchwork)
library(scales)

# Set seed for reproducibility
set.seed(42)

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

# Read the data
data <- read.csv("all_tables.csv", stringsAsFactors = FALSE)

# Filter for HOTELLING model only
hotelling_data <- data %>% filter(Model == "HOTELLING")

# Create long format for both agents
hotelling_long <- hotelling_data %>%
  pivot_longer(
    cols = c(Agent1_Delta, Agent2_Delta, Agent1_RPDI, Agent2_RPDI, 
             Agent1_Avg_Prices, Agent2_Avg_Prices),
    names_to = c("Agent", ".value"),
    names_pattern = "(Agent[12])_(.*)"
  ) %>%
  rename(Delta = Delta, RPDI = RPDI, Avg_Prices = Avg_Prices)

# Extract algorithm from Matchup
hotelling_long <- hotelling_long %>%
  mutate(
    Algorithm = case_when(
      Agent == "Agent1" ~ str_extract(Matchup, "^[A-Z]+"),
      Agent == "Agent2" ~ str_extract(Matchup, "vs ([A-Z]+)$", group = 1)
    ),
    Algorithm = case_when(
      Algorithm == "Q" ~ "Q-learning",
      Algorithm == "DQN" ~ "DQN",
      Algorithm == "PSO" ~ "PSO",
      Algorithm == "DDPG" ~ "DDPG",
      TRUE ~ Algorithm
    ),
    Shock_Condition = case_when(
      Shock == "0" ~ "No Shock",
      Shock == "A" ~ "Shock A",
      Shock == "B" ~ "Shock B",
      Shock == "C" ~ "Shock C"
    ),
    Shock_Condition = factor(Shock_Condition, 
                              levels = c("No Shock", "Shock A", "Shock B", "Shock C"))
  )

# =============================================================================
# DEFINE COMMON AESTHETICS
# =============================================================================

# Color palette for shock conditions (colorblind-friendly)
shock_colors <- c("No Shock" = "#E41A1C",   # Red
                  "Shock A" = "#377EB8",     # Blue
                  "Shock B" = "#4DAF4A",     # Green
                  "Shock C" = "#984EA3")     # Purple

# Shape palette for algorithms
algo_shapes <- c("Q-learning" = 16,  # Circle (filled)
                 "DQN" = 15,          # Square (filled)
                 "PSO" = 17,          # Triangle (filled)
                 "DDPG" = 18)         # Diamond (filled)

# Common theme for publication quality
theme_publication <- theme_bw(base_size = 11) +
  theme(
    plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 9),
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 9),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
    plot.margin = margin(5, 5, 5, 5)
  )

# =============================================================================
# PANEL 1: DELTA vs RPDI SCATTER PLOT
# =============================================================================

p1_scatter <- ggplot(hotelling_long, aes(x = RPDI, y = Delta)) +
  # Reference diagonal line (Delta = RPDI)
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", 
              color = "gray40", linewidth = 0.7) +
  # Data points
  geom_point(aes(shape = Algorithm, color = Shock_Condition), 
             size = 3.5, alpha = 0.85, stroke = 0.5) +
  # Scales
  scale_color_manual(values = shock_colors) +
  scale_shape_manual(values = algo_shapes) +
  # Labels
  labs(
    title = "Δ vs RPDI Relationship",
    x = "Relative Price Difference Index (RPDI)",
    y = "Delta (Δ)"
  ) +
  # Add annotation for the diagonal line
  annotate("text", x = 0.35, y = 0.39, label = "Delta = RPDI Line", 
           angle = 45, size = 3, color = "gray40", fontface = "italic") +
  # Theme
  theme_publication +
  theme(legend.position = "none")

# =============================================================================
# PANEL 2: SHOCK IMPACT LINE PLOT
# =============================================================================

# Aggregate data by algorithm and shock condition
shock_impact_data <- hotelling_long %>%
  group_by(Algorithm, Shock_Condition) %>%
  summarise(
    Mean_Delta = mean(Delta, na.rm = TRUE),
    .groups = "drop"
  )

p2_shock <- ggplot(shock_impact_data, 
                   aes(x = Shock_Condition, y = Mean_Delta, 
                       color = Algorithm, group = Algorithm)) +
  # Reference lines
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.6) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red", linewidth = 0.6) +
  # Shaded regions (sub-Nash in pink - though Hotelling stays above 0)
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0,
           fill = "#FFCCCC", alpha = 0.3) +
  # Lines and points
  geom_line(linewidth = 1.2) +
  geom_point(aes(shape = Algorithm), size = 3.5, fill = "white") +
  # Scales
  scale_color_manual(values = c("Q-learning" = "#1B9E77", 
                                 "DQN" = "#D95F02",
                                 "PSO" = "#7570B3", 
                                 "DDPG" = "#E7298A")) +
  scale_shape_manual(values = algo_shapes) +
  scale_y_continuous(limits = c(-0.2, 1.2)) +
  # Labels
  labs(
    title = "Shock Impact on Δ",
    x = "Shock Condition",
    y = "Delta (Δ)"
  ) +
  # Annotations
  annotate("text", x = 0.6, y = 0.08, label = "Nash (Δ=0)", 
           size = 2.8, color = "black", hjust = 0) +
  annotate("text", x = 0.6, y = 1.08, label = "Monopoly (Δ=1)", 
           size = 2.8, color = "red", hjust = 0) +
  # Add annotation highlighting stability
  annotate("text", x = 2.5, y = 0.9, 
           label = "Remarkable\nStability", 
           size = 3, color = "#1B9E77", fontface = "italic") +
  # Theme
  theme_publication +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# =============================================================================
# PANEL 3: PRICE STABILITY VS BENCHMARK SHIFT
# =============================================================================

# Prepare price data
price_data <- hotelling_long %>%
  group_by(Algorithm, Shock_Condition) %>%
  summarise(
    Mean_Price = mean(Avg_Prices, na.rm = TRUE),
    Theo_Price = mean(Theo_Prices, na.rm = TRUE),
    .groups = "drop"
  )

# Theoretical Nash prices for HOTELLING (constant = 1.0)
theo_nash <- data.frame(
  Shock_Condition = factor(c("No Shock", "Shock A", "Shock B", "Shock C"),
                           levels = c("No Shock", "Shock A", "Shock B", "Shock C")),
  Nash_Price = c(1.0, 1.0, 1.0, 1.0)
)

p3_price <- ggplot() +
  # Theoretical Nash line (thick dashed)
  geom_line(data = theo_nash, 
            aes(x = Shock_Condition, y = Nash_Price, group = 1),
            linetype = "dashed", linewidth = 1.5, color = "black") +
  geom_point(data = theo_nash,
             aes(x = Shock_Condition, y = Nash_Price),
             size = 4, shape = 4, stroke = 1.5, color = "black") +
  # Algorithm actual prices
  geom_line(data = price_data, 
            aes(x = Shock_Condition, y = Mean_Price, 
                color = Algorithm, group = Algorithm),
            linewidth = 1.0) +
  geom_point(data = price_data,
             aes(x = Shock_Condition, y = Mean_Price, 
                 color = Algorithm, shape = Algorithm),
             size = 3) +
  # Scales
  scale_color_manual(values = c("Q-learning" = "#1B9E77", 
                                 "DQN" = "#D95F02",
                                 "PSO" = "#7570B3", 
                                 "DDPG" = "#E7298A")) +
  scale_shape_manual(values = algo_shapes) +
  scale_y_continuous(limits = c(0.95, 1.25)) +
  # Labels
  labs(
    title = "Price vs Benchmark",
    x = "Shock Condition",
    y = "Price"
  ) +
  # Annotations
  annotate("text", x = 1, y = 0.97, label = "Theoretical Nash (p=1.0)", 
           size = 2.8, color = "black", fontface = "italic") +
  annotate("text", x = 2.5, y = 1.22, 
           label = "Prices stable\nabove Nash", 
           size = 2.8, color = "#D95F02", fontface = "italic") +
  # Theme
  theme_publication +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# =============================================================================
# CREATE SHARED LEGEND
# =============================================================================

# Create a dummy plot for extracting legends
legend_data <- hotelling_long %>%
  select(Algorithm, Shock_Condition, Delta, RPDI) %>%
  distinct()

p_legend <- ggplot(legend_data, aes(x = RPDI, y = Delta)) +
  geom_point(aes(shape = Algorithm), size = 4, color = "black") +
  geom_point(aes(color = Shock_Condition), size = 4, shape = 16) +
  scale_color_manual(values = shock_colors, name = "Shock Condition") +
  scale_shape_manual(values = algo_shapes, name = "Algorithm") +
  guides(
    shape = guide_legend(order = 1, title.position = "top"),
    color = guide_legend(order = 2, title.position = "top")
  ) +
  theme_publication +
  theme(
    legend.box = "vertical",
    legend.spacing.y = unit(0.5, "cm"),
    legend.title = element_text(size = 11, face = "bold"),
    legend.text = element_text(size = 10)
  )

# Extract legend using cowplot
library(cowplot)
legend_grob <- get_legend(p_legend)

# =============================================================================
# COMBINE ALL PANELS WITH SHARED LEGEND
# =============================================================================

# Combine the three plots
combined_plots <- p1_scatter + p2_shock + p3_price +
  plot_layout(ncol = 3, widths = c(1, 1, 1))

# Final assembly with legend on the right
final_figure <- plot_grid(
  combined_plots,
  legend_grob,
  ncol = 2,
  rel_widths = c(3, 0.4)
)

# Add overall title
title <- ggdraw() + 
  draw_label(
    "HOTELLING Model: Algorithmic Competition Analysis",
    fontface = 'bold',
    size = 14,
    x = 0.5,
    hjust = 0.5
  )

# Final plot with title
final_plot <- plot_grid(
  title,
  final_figure,
  ncol = 1,
  rel_heights = c(0.05, 1)
)

# =============================================================================
# SAVE THE FIGURE
# =============================================================================

# Save as high-resolution PNG
ggsave("HOTELLING_comprehensive_figure.png", 
       plot = final_plot,
       width = 14, 
       height = 5,
       dpi = 600,
       bg = "white")

# Save as PDF (vector format for publication)
ggsave("HOTELLING_comprehensive_figure.pdf", 
       plot = final_plot,
       width = 14, 
       height = 5,
       device = cairo_pdf)

# Display message
cat("HOTELLING comprehensive figure saved successfully!\n")
cat("Files created: HOTELLING_comprehensive_figure.png, HOTELLING_comprehensive_figure.pdf\n")
