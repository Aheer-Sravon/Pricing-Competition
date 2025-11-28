# =============================================================================
# FIGURE 3: Price Stability vs Benchmark Shift - LOGIT Model Only
# =============================================================================
# Single panel showing actual algorithm prices vs theoretical Nash prices
# Explains counterintuitive negative indicators despite supra-competitive pricing
# With legend on the right
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

# Filter for LOGIT model only
logit_data <- data %>% filter(Model == "LOGIT")

# Create long format for both agents
logit_long <- logit_data %>%
  pivot_longer(
    cols = c(Agent1_Delta, Agent2_Delta, Agent1_RPDI, Agent2_RPDI, 
             Agent1_Avg_Prices, Agent2_Avg_Prices),
    names_to = c("Agent", ".value"),
    names_pattern = "(Agent[12])_(.*)"
  ) %>%
  rename(Delta = Delta, RPDI = RPDI, Avg_Prices = Avg_Prices)

# Extract algorithm from Matchup
logit_long <- logit_long %>%
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
# PREPARE PRICE DATA
# =============================================================================

# Aggregate price data by algorithm and shock condition
price_data <- logit_long %>%
  group_by(Algorithm, Shock_Condition) %>%
  summarise(
    Mean_Price = mean(Avg_Prices, na.rm = TRUE),
    .groups = "drop"
  )

# Theoretical Nash prices for LOGIT
theo_nash <- data.frame(
  Shock_Condition = factor(c("No Shock", "Shock A", "Shock B", "Shock C"),
                           levels = c("No Shock", "Shock A", "Shock B", "Shock C")),
  Nash_Price = c(1.47, 1.80, 1.54, 1.91)
)

# =============================================================================
# DEFINE AESTHETICS
# =============================================================================

# Color palette for algorithms
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
theme_publication <- theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray40"),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 10),
    legend.title = element_text(size = 11, face = "bold"),
    legend.text = element_text(size = 10),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
    plot.margin = margin(10, 10, 10, 10)
  )

# =============================================================================
# CREATE MAIN PLOT
# =============================================================================

p_main <- ggplot() +
  # Shaded region between actual prices and Nash under Shock C
  annotate("rect", xmin = 3.7, xmax = 4.3, ymin = 1.55, ymax = 1.91,
           fill = "#FFE4E1", alpha = 0.8) +
  
  # Theoretical Nash line (thick dashed black)
  geom_line(data = theo_nash, 
            aes(x = Shock_Condition, y = Nash_Price, group = 1),
            linetype = "dashed", linewidth = 1.8, color = "black") +
  geom_point(data = theo_nash,
             aes(x = Shock_Condition, y = Nash_Price),
             size = 5, shape = 4, stroke = 2, color = "black") +
  
  # Algorithm actual prices - lines
  geom_line(data = price_data, 
            aes(x = Shock_Condition, y = Mean_Price, 
                color = Algorithm, group = Algorithm),
            linewidth = 1.2) +
  
  # Algorithm actual prices - points
  geom_point(data = price_data,
             aes(x = Shock_Condition, y = Mean_Price, 
                 color = Algorithm, shape = Algorithm),
             size = 4) +
  
  # Scales
  scale_color_manual(values = algo_colors, name = "Algorithm") +
  scale_shape_manual(values = algo_shapes, name = "Algorithm") +
  scale_y_continuous(limits = c(1.35, 2.05), 
                     breaks = seq(1.4, 2.0, by = 0.1)) +
  
  # Labels
  labs(
    title = "LOGIT Model: Price Stability vs Benchmark Shift",
    subtitle = "Algorithms maintain stable prices while Nash equilibrium shifts upward",
    x = "Shock Condition",
    y = "Price"
  ) +
  
  # Annotation: Gap arrow under Shock C
  annotate("segment", x = 4.15, xend = 4.15, y = 1.58, yend = 1.88,
           arrow = arrow(ends = "both", length = unit(0.15, "inches")),
           color = "#E41A1C", linewidth = 1) +
  annotate("text", x = 4.35, y = 1.73, label = "Gap ≈ 0.25", 
           size = 3.5, color = "#E41A1C", fontface = "bold", hjust = 0) +
  
  # Annotation: Theoretical Nash label
  annotate("text", x = 1.3, y = 1.98, label = "Theoretical Nash", 
           size = 3.5, color = "black", fontface = "bold") +
  annotate("segment", x = 1.15, xend = 1.0, y = 1.95, yend = 1.82,
           arrow = arrow(length = unit(0.1, "inches")),
           color = "black", linewidth = 0.5) +
  
  # Annotation: Algorithm prices label
  annotate("text", x = 2.7, y = 1.45, 
           label = "Algorithm prices\nremain near ~1.65", 
           size = 3.2, color = "#1B9E77", fontface = "italic") +
  
  # Annotation: Original no-shock Nash reference
  geom_hline(yintercept = 1.47, linetype = "dotted", 
             color = "gray50", linewidth = 0.8) +
  annotate("text", x = 4.5, y = 1.44, 
           label = "No-shock Nash (1.47)", 
           size = 2.8, color = "gray50", hjust = 1) +
  
  # Annotation box explaining the key insight
 annotate("label", x = 2.5, y = 2.02, 
           label = "Key: Prices stay above original Nash (1.47)\nbut below shifted Nash → Negative RPDI",
           size = 3, fill = "#FFFACD", color = "gray30",
           label.padding = unit(0.4, "lines"),
           label.r = unit(0.15, "lines")) +
  
  # Theme
  theme_publication +
  theme(
    legend.position = "right",
    legend.box = "vertical",
    axis.text.x = element_text(size = 10)
  ) +
  guides(
    color = guide_legend(title.position = "top", order = 1),
    shape = guide_legend(title.position = "top", order = 1)
  )

# =============================================================================
# SAVE THE FIGURE
# =============================================================================

# Save as high-resolution PNG
ggsave("Figure3_Price_vs_Benchmark_LOGIT.png", 
       plot = p_main,
       width = 10, 
       height = 6,
       dpi = 600,
       bg = "white")

# Save as PDF (vector format for publication)
ggsave("Figure3_Price_vs_Benchmark_LOGIT.pdf", 
       plot = p_main,
       width = 10, 
       height = 6,
       device = cairo_pdf)

# Display message
cat("Figure 3: Price vs Benchmark (LOGIT) saved successfully!\n")
cat("Files created: Figure3_Price_vs_Benchmark_LOGIT.png, Figure3_Price_vs_Benchmark_LOGIT.pdf\n")
