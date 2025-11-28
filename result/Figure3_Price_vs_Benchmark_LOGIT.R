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

# Filter for LOGIT model
logit_data <- data %>% filter(Model == "LOGIT")

# Create long format
logit_long <- logit_data %>%
  pivot_longer(
    cols = c(Agent1_Delta, Agent2_Delta, Agent1_RPDI, Agent2_RPDI, 
             Agent1_Avg_Prices, Agent2_Avg_Prices),
    names_to = c("Agent", ".value"),
    names_pattern = "(Agent[12])_(.*)"
  ) %>%
  rename(Delta = Delta, RPDI = RPDI, Avg_Prices = Avg_Prices)

# Extract algorithm
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
    Algorithm = factor(Algorithm, levels = c("Q-learning", "DQN", "PSO", "DDPG")),
    Shock_Condition = case_when(
      Shock == "0" ~ "No Shock",
      Shock == "A" ~ "Shock A",
      Shock == "B" ~ "Shock B",
      Shock == "C" ~ "Shock C"
    ),
    Shock_Condition = factor(Shock_Condition, 
                              levels = c("No Shock", "Shock A", "Shock B", "Shock C"))
  )

# PREPARE PRICE DATA
price_data <- logit_long %>%
  group_by(Algorithm, Shock_Condition) %>%
  summarise(
    Mean_Price = mean(Avg_Prices, na.rm = TRUE),
    SE_Price = sd(Avg_Prices, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

# Theoretical Nash prices for LOGIT
theo_nash <- data.frame(
  Shock_Condition = factor(c("No Shock", "Shock A", "Shock B", "Shock C"),
                           levels = c("No Shock", "Shock A", "Shock B", "Shock C")),
  Nash_Price = c(1.47, 1.80, 1.54, 1.91)
)

# Q1 JOURNAL COLOR PALETTE (Colorblind-friendly)
algo_colors <- c(
  "Q-learning" = "#1B9E77",
  "PSO"        = "#7570B3", 
  "DQN"        = "#D95F02",
  "DDPG"       = "#E7298A"
)

algo_shapes <- c(
  "Q-learning" = 16,
  "PSO"        = 17,
  "DQN"        = 15,
  "DDPG"       = 18
)

algo_linetypes <- c(
  "Q-learning" = "solid",
  "PSO"        = "solid",
  "DQN"        = "solid",
  "DDPG"       = "solid"
)

# THEME (Cross-platform compatible)
theme_q1_journal <- function(base_size = 11) {
  theme_bw(base_size = base_size) +
    theme(
      text = element_text(family = 'serif'),
      plot.title = element_text(
        size = rel(1.2), 
        face = "bold", 
        hjust = 0.5,
        margin = margin(b = 6)
      ),
      plot.subtitle = element_text(
        size = rel(0.9),
        hjust = 0.5,
        margin = margin(b = 10),
        color = "gray35",
        face = "italic"
      ),
      axis.title = element_text(size = rel(1.0), face = "plain"),
      axis.title.x = element_text(margin = margin(t = 10)),
      axis.title.y = element_text(margin = margin(r = 10)),
      axis.text = element_text(size = rel(0.95), color = "black"),
      panel.grid.major = element_line(color = "gray88", linewidth = 0.35),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.6),
      legend.title = element_text(size = rel(0.95), face = "bold"),
      legend.text = element_text(size = rel(0.9)),
      legend.key.size = unit(1.1, "lines"),
      legend.key.width = unit(2.0, "lines"),
      legend.background = element_rect(fill = "white", color = "gray80", linewidth = 0.3),
      legend.key = element_rect(fill = "white", color = NA),
      legend.position = "right",
      legend.box = "vertical",
      legend.margin = margin(t = 5, r = 5, b = 5, l = 5),
      plot.margin = margin(t = 12, r = 12, b = 12, l = 12)
    )
}

# CREATE MAIN FIGURE
p_main <- ggplot() +
  
  # Shaded region: Gap between actual and Nash under Shock C
  annotate("rect", xmin = 3.75, xmax = 4.25, ymin = 1.58, ymax = 1.88,
           fill = "#FFE4E1", alpha = 0.7) +
  
  # Reference: No-Shock Nash horizontal line
  geom_hline(yintercept = 1.47, linetype = "dotted", 
             color = "gray55", linewidth = 0.6) +
  
  # Theoretical Nash prices (Black dashed line with X markers)
  geom_line(data = theo_nash, 
            aes(x = Shock_Condition, y = Nash_Price, group = 1),
            linetype = "longdash", linewidth = 1.3, color = "black") +
  geom_point(data = theo_nash,
             aes(x = Shock_Condition, y = Nash_Price),
             size = 4, shape = 4, stroke = 1.8, color = "black") +
  
  # Algorithm actual prices
  geom_line(data = price_data, 
            aes(x = Shock_Condition, y = Mean_Price, 
                color = Algorithm, group = Algorithm, linetype = Algorithm),
            linewidth = 0.95) +
  geom_point(data = price_data,
             aes(x = Shock_Condition, y = Mean_Price, 
                 color = Algorithm, shape = Algorithm),
             size = 3, stroke = 0.4) +
  
  # Scales
  scale_color_manual(values = algo_colors, name = "Algorithm") +
  scale_shape_manual(values = algo_shapes, name = "Algorithm") +
  scale_linetype_manual(values = algo_linetypes, name = "Algorithm") +
  scale_y_continuous(
    limits = c(1.38, 2.00),
    breaks = seq(1.4, 2.0, by = 0.1),
    minor_breaks = NULL,
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  
  # Labels
  labs(
    title = "Logit Model: Price Stability vs Shifting Benchmark",
    x = "Shock Condition",
    y = "Price"
  ) +
  
  # Theoretical Nash label
  annotate("text", x = 2.3, y = 1.8, 
           label = "Theoretical Nash", 
           size = 3.0, color = "black", hjust = 0.5,
           lineheight = 0.85, angle = -50) +
  
  # No-shock Nash reference label
  annotate("text", x = 4.5, y = 1.445, 
           label = "No-shock Nash", 
           size = 2.6, color = "gray55", hjust = 1) +
  
  # Theme
  theme_q1_journal() +
  guides(
    color = guide_legend(order = 1, override.aes = list(linewidth = 1.2)),
    shape = guide_legend(order = 1),
    linetype = guide_legend(order = 1)
  )

# SAVE FIGURES
ggsave(
  filename = "./figures/Figure3_Price_vs_Benchmark_LOGIT.png",
  plot = p_main,
  width = 160,
  height = 100,
  units = "mm",
  dpi = 600,
  bg = "white"
)

ggsave(
  filename = "./figures/Figure3_Price_vs_Benchmark_LOGIT.pdf",
  plot = p_main,
  width = 160,
  height = 100,
  units = "mm",
  device = "pdf"
)

cat("\n")
cat("============================================================\n")
cat("  FIGURE 3: Price Stability vs Benchmark Shift (LOGIT)\n")
cat("  Q1 JOURNAL PUBLICATION STANDARD\n")
cat("============================================================\n")
cat("\n")
cat("Files created:\n")
cat("  - Figure3_Price_vs_Benchmark_LOGIT.png  (600 DPI)\n")
cat("  - Figure3_Price_vs_Benchmark_LOGIT.pdf  (Vector)\n")
# cat("  - Figure3_Price_vs_Benchmark_LOGIT.tiff (600 DPI, LZW)\n")
cat("\n")
cat("Key visual elements:\n")
cat("  - Black dashed line: Theoretical Nash prices (1.47 -> 1.91)\n")
cat("  - Colored lines: Algorithm actual prices (~1.55-1.70)\n")
cat("  - Pink shaded region: Gap between actual and Nash under Shock C\n")
cat("  - Dotted horizontal: No-shock Nash reference (1.47)\n")
cat("\n")
cat("Key insight communicated:\n")
cat("  Algorithms maintain stable prices above the original Nash (1.47)\n")
cat("  but below the shifted Nash (1.91), creating negative RPDI values.\n")
cat("\n")
