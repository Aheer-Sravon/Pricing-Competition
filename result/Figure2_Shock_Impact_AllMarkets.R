# =============================================================================
# FIGURE 2: Shock Impact Across Market Structures
# =============================================================================
# Three panels side-by-side: LOGIT | HOTELLING | LINEAR
# Shows divergent responses to demand shocks across market structures
# With shared legend on the right
# =============================================================================

# Load required libraries
library(tidyverse)
library(ggplot2)
library(patchwork)
library(scales)
library(cowplot)

# Set seed for reproducibility
set.seed(42)

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

# Read the data
data <- read.csv("all_tables.csv", stringsAsFactors = FALSE)

# Create long format for both agents
data_long <- data %>%
  pivot_longer(
    cols = c(Agent1_Delta, Agent2_Delta, Agent1_RPDI, Agent2_RPDI, 
             Agent1_Avg_Prices, Agent2_Avg_Prices),
    names_to = c("Agent", ".value"),
    names_pattern = "(Agent[12])_(.*)"
  ) %>%
  rename(Delta = Delta, RPDI = RPDI, Avg_Prices = Avg_Prices)

# Extract algorithm from Matchup
data_long <- data_long %>%
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
                              levels = c("No Shock", "Shock A", "Shock B", "Shock C")),
    Model = factor(Model, levels = c("LOGIT", "HOTELLING", "LINEAR"))
  )

# Remove extreme outliers
data_clean <- data_long %>%
  filter(Delta > -8 & Delta < 3)

# =============================================================================
# DEFINE COMMON AESTHETICS
# =============================================================================

# Color palette for algorithms (consistent with your style)
algo_colors <- c("Q-learning" = "#1B9E77",
                 "DQN" = "#D95F02",
                 "PSO" = "#7570B3", 
                 "DDPG" = "#E7298A")

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
# PANEL 1: LOGIT - Shock Impact
# =============================================================================

logit_shock <- data_clean %>% 
  filter(Model == "LOGIT") %>%
  group_by(Algorithm, Shock_Condition) %>%
  summarise(
    Mean_Delta = mean(Delta, na.rm = TRUE),
    .groups = "drop"
  )

p1_logit <- ggplot(logit_shock, 
                   aes(x = Shock_Condition, y = Mean_Delta, 
                       color = Algorithm, group = Algorithm)) +
  # Shaded region for sub-Nash performance
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0,
           fill = "#FFCCCC", alpha = 0.3) +
  # Reference lines
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.6) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red", linewidth = 0.6) +
  # Lines and points
  geom_line(linewidth = 1.2) +
  geom_point(aes(shape = Algorithm), size = 3.5) +
  # Scales
  scale_color_manual(values = algo_colors) +
  scale_shape_manual(values = algo_shapes) +
  scale_y_continuous(limits = c(-7, 1.5), breaks = seq(-6, 1, by = 1)) +
  # Labels
  labs(
    title = "LOGIT Model",
    subtitle = "Catastrophic Collapse",
    x = "Shock Condition",
    y = "Delta (Δ)"
  ) +
  # Annotations
  annotate("text", x = 0.55, y = 0.2, label = "Nash", 
           size = 2.5, color = "black", hjust = 0, fontface = "italic") +
  annotate("text", x = 0.55, y = 1.2, label = "Monopoly", 
           size = 2.5, color = "red", hjust = 0, fontface = "italic") +
  annotate("text", x = 3, y = -5.5, label = "All algorithms\ncollapse", 
           size = 2.8, color = "#E41A1C", fontface = "bold") +
  # Theme
  theme_publication +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.subtitle = element_text(size = 10, hjust = 0.5, color = "#E41A1C")
  )

# =============================================================================
# PANEL 2: HOTELLING - Shock Impact
# =============================================================================

hotelling_shock <- data_clean %>% 
  filter(Model == "HOTELLING") %>%
  group_by(Algorithm, Shock_Condition) %>%
  summarise(
    Mean_Delta = mean(Delta, na.rm = TRUE),
    .groups = "drop"
  )

p2_hotelling <- ggplot(hotelling_shock, 
                       aes(x = Shock_Condition, y = Mean_Delta, 
                           color = Algorithm, group = Algorithm)) +
  # Shaded region for sub-Nash performance
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0,
           fill = "#FFCCCC", alpha = 0.3) +
  # Reference lines
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.6) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red", linewidth = 0.6) +
  # Lines and points
  geom_line(linewidth = 1.2) +
  geom_point(aes(shape = Algorithm), size = 3.5) +
  # Scales
  scale_color_manual(values = algo_colors) +
  scale_shape_manual(values = algo_shapes) +
  scale_y_continuous(limits = c(-0.3, 1.2), breaks = seq(0, 1, by = 0.25)) +
  # Labels
  labs(
    title = "HOTELLING Model",
    subtitle = "Remarkable Stability",
    x = "Shock Condition",
    y = "Delta (Δ)"
  ) +
  # Annotations
  annotate("text", x = 0.55, y = 0.08, label = "Nash", 
           size = 2.5, color = "black", hjust = 0, fontface = "italic") +
  annotate("text", x = 0.55, y = 1.08, label = "Monopoly", 
           size = 2.5, color = "red", hjust = 0, fontface = "italic") +
  annotate("text", x = 2.5, y = 0.85, label = "Lines remain\nnearly flat", 
           size = 2.8, color = "#1B9E77", fontface = "bold") +
  # Theme
  theme_publication +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.subtitle = element_text(size = 10, hjust = 0.5, color = "#4DAF4A")
  )

# =============================================================================
# PANEL 3: LINEAR - Shock Impact
# =============================================================================

linear_shock <- data_clean %>% 
  filter(Model == "LINEAR") %>%
  group_by(Algorithm, Shock_Condition) %>%
  summarise(
    Mean_Delta = mean(Delta, na.rm = TRUE),
    .groups = "drop"
  )

p3_linear <- ggplot(linear_shock, 
                    aes(x = Shock_Condition, y = Mean_Delta, 
                        color = Algorithm, group = Algorithm)) +
  # Shaded regions
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0,
           fill = "#FFCCCC", alpha = 0.3) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = 1, ymax = Inf,
           fill = "#FFFFCC", alpha = 0.3) +
  # Reference lines
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.6) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red", linewidth = 0.6) +
  # Lines and points
  geom_line(linewidth = 1.2) +
  geom_point(aes(shape = Algorithm), size = 3.5) +
  # Scales
  scale_color_manual(values = algo_colors) +
  scale_shape_manual(values = algo_shapes) +
  scale_y_continuous(limits = c(-0.3, 1.6), breaks = seq(0, 1.5, by = 0.5)) +
  # Labels
  labs(
    title = "LINEAR Model",
    subtitle = "Profit Inflation",
    x = "Shock Condition",
    y = "Delta (Δ)"
  ) +
  # Annotations
  annotate("text", x = 0.55, y = 0.08, label = "Nash", 
           size = 2.5, color = "black", hjust = 0, fontface = "italic") +
  annotate("text", x = 0.55, y = 1.08, label = "Monopoly", 
           size = 2.5, color = "red", hjust = 0, fontface = "italic") +
  annotate("text", x = 3.5, y = 1.45, label = "Supra-\nMonopoly", 
           size = 2.8, color = "#984EA3", fontface = "bold") +
  # Theme
  theme_publication +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.subtitle = element_text(size = 10, hjust = 0.5, color = "#984EA3")
  )

# =============================================================================
# CREATE SHARED LEGEND
# =============================================================================

# Create a dummy plot for extracting legend
p_legend <- ggplot(data_clean, aes(x = Shock_Condition, y = Delta)) +
  geom_line(aes(color = Algorithm, group = Algorithm), linewidth = 1.2) +
  geom_point(aes(color = Algorithm, shape = Algorithm), size = 4) +
  scale_color_manual(values = algo_colors, name = "Algorithm") +
  scale_shape_manual(values = algo_shapes, name = "Algorithm") +
  guides(
    color = guide_legend(title.position = "top"),
    shape = guide_legend(title.position = "top")
  ) +
  theme_publication +
  theme(
    legend.box = "vertical",
    legend.spacing.y = unit(0.3, "cm"),
    legend.title = element_text(size = 11, face = "bold"),
    legend.text = element_text(size = 10)
  )

# Extract legend
legend_grob <- get_legend(p_legend)

# =============================================================================
# COMBINE ALL PANELS WITH SHARED LEGEND
# =============================================================================

# Combine the three plots
combined_plots <- p1_logit + p2_hotelling + p3_linear +
  plot_layout(ncol = 3, widths = c(1, 1, 1))

# Final assembly with legend on the right
final_figure <- plot_grid(
  combined_plots,
  legend_grob,
  ncol = 2,
  rel_widths = c(3, 0.35)
)

# Add overall title
title <- ggdraw() + 
  draw_label(
    "Shock Impact Across Market Structures",
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
ggsave("./figures/Figure2_Shock_Impact_AllMarkets.png", 
       plot = final_plot,
       width = 14, 
       height = 5,
       dpi = 600,
       bg = "white")

# Save as PDF (vector format for publication)
ggsave("./figures/Figure2_Shock_Impact_AllMarkets.pdf", 
       plot = final_plot,
       width = 14, 
       height = 5,
       device = cairo_pdf)

# Display message
cat("Figure 2: Shock Impact Across Market Structures saved successfully!\n")
cat("Files created: Figure2_Shock_Impact_AllMarkets.png, Figure2_Shock_Impact_AllMarkets.pdf\n")
