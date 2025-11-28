# =============================================================================
# FIGURE 1: Delta vs RPDI Scatter Plot - All Market Structures
# =============================================================================
# Three panels side-by-side: LOGIT | HOTELLING | LINEAR
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

# Remove extreme outliers for visualization
data_clean <- data_long %>%
  filter(Delta > -8 & Delta < 3, RPDI > -2 & RPDI < 1.5)

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
# PANEL 1: LOGIT - Delta vs RPDI
# =============================================================================

logit_data <- data_clean %>% filter(Model == "LOGIT")

p1_logit <- ggplot(logit_data, aes(x = RPDI, y = Delta)) +
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
    title = "LOGIT Model",
    x = "Relative Price Difference Index (RPDI)",
    y = "Delta (Δ)"
  ) +
  # Add annotation for the diagonal line
  annotate("text", x = -0.5, y = 0, label = "Δ = RPDI", 
           angle = 32, size = 3, color = "gray40", fontface = "italic") +
  # Theme
  theme_publication +
  theme(legend.position = "none")

# =============================================================================
# PANEL 2: HOTELLING - Delta vs RPDI
# =============================================================================

hotelling_data <- data_clean %>% filter(Model == "HOTELLING")

p2_hotelling <- ggplot(hotelling_data, aes(x = RPDI, y = Delta)) +
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
    title = "HOTELLING Model",
    x = "Relative Price Difference Index (RPDI)",
    y = "Delta (Δ)"
  ) +
  # Add annotation for the diagonal line
  annotate("text", x = 0.35, y = 0.5, label = "Δ = RPDI", 
           angle = 40, size = 3, color = "gray40", fontface = "italic") +
  # Theme
  theme_publication +
  theme(legend.position = "none")

# =============================================================================
# PANEL 3: LINEAR - Delta vs RPDI
# =============================================================================

linear_data <- data_clean %>% filter(Model == "LINEAR")

p3_linear <- ggplot(linear_data, aes(x = RPDI, y = Delta)) +
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
    title = "LINEAR Model",
    x = "Relative Price Difference Index (RPDI)",
    y = "Delta (Δ)"
  ) +
  # Add annotation for the diagonal line
  annotate("text", x = 0.15, y = 0.35, label = "Δ = RPDI", 
           angle = 38, size = 3, color = "gray40", fontface = "italic") +
  # Highlight the decoupling region
  annotate("rect", xmin = 0.4, xmax = 0.9, ymin = -0.2, ymax = 0.2,
           fill = "yellow", alpha = 0.2) +
  annotate("text", x = 0.65, y = -0.35, 
           label = "Price-Profit\nDecoupling", 
           size = 2.8, color = "#D95F02", fontface = "bold") +
  # Theme
  theme_publication +
  theme(legend.position = "none")

# =============================================================================
# CREATE SHARED LEGEND
# =============================================================================

# Create a dummy plot for extracting legends
p_legend <- ggplot(data_clean, aes(x = RPDI, y = Delta)) +
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
  rel_widths = c(3, 0.4)
)

# Add overall title
title <- ggdraw() + 
  draw_label(
    "Relationship between Profit Extraction (Delta) and Price Elevation (RPDI)",
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
ggsave("./figures/Figure1_Delta_vs_RPDI_AllMarkets.png", 
       plot = final_plot,
       width = 14, 
       height = 5,
       dpi = 600,
       bg = "white")

# Save as PDF (vector format for publication)
ggsave("./figures/Figure1_Delta_vs_RPDI_AllMarkets.pdf", 
       plot = final_plot,
       width = 14, 
       height = 5,
       device = cairo_pdf)

# Display message
cat("Figure 1: Delta vs RPDI (All Markets) saved successfully!\n")
cat("Files created: Figure1_Delta_vs_RPDI_AllMarkets.png, Figure1_Delta_vs_RPDI_AllMarkets.pdf\n")
