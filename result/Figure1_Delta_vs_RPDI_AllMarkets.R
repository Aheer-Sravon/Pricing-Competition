library(tidyverse)
library(ggplot2)
library(patchwork)
library(scales)
library(cowplot)

POINT_SIZE <- 1.3
STROKE_SIZE <- 0.8

set.seed(42)

# Read the data
data <- read.csv("./all_tables.csv", stringsAsFactors = FALSE)

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

# COLOR PALETTE (Colorblind-friendly - Okabe-Ito inspired)
shock_colors <- c(
  "No Shock" = "#D55E00",
  "Shock A"  = "#0072B2",
  "Shock B"  = "#009E73",
  "Shock C"  = "#CC79A7"
)

algo_shapes <- c(
  "Q-learning" = 21,
  "PSO"        = 24,
  "DQN"        = 22,
  "DDPG"       = 23
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
        margin = margin(b = 8)
      ),
      axis.title = element_text(size = rel(1.0), face = "plain"),
      axis.title.x = element_text(margin = margin(t = 8)),
      axis.title.y = element_text(margin = margin(r = 8)),
      axis.text = element_text(size = rel(0.9), color = "black"),
      panel.grid.major = element_line(color = "gray85", linewidth = 0.4),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.6),
      legend.title = element_text(size = rel(0.95), face = "bold"),
      legend.text = element_text(size = rel(0.85)),
      legend.key.size = unit(0.9, "lines"),
      legend.background = element_rect(fill = "white", color = NA),
      legend.key = element_rect(fill = "white", color = NA),
      plot.margin = margin(t = 10, r = 10, b = 10, l = 10),
      strip.background = element_rect(fill = "gray95", color = "black", linewidth = 0.5),
      strip.text = element_text(size = rel(1.0), face = "bold")
    )
}

# PANEL 1: LOGIT - Delta vs RPDI
logit_data <- data_clean |> filter(Model == "LOGIT")

p1_logit <- ggplot(logit_data, aes(x = RPDI, y = Delta)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", 
              color = "gray50", linewidth = 0.6) +
  geom_point(aes(shape = Algorithm, color = Shock_Condition), 
             size = POINT_SIZE, alpha = 0.9, stroke = STROKE_SIZE) +
  scale_color_manual(values = shock_colors) +
  scale_shape_manual(values = algo_shapes) +
  scale_x_continuous(breaks = seq(-1.5, 1, by = 0.5)) +
  scale_y_continuous(breaks = seq(-7, 2, by = 1)) +
  labs(
    title = "Logit",
    x = "RPDI",
    y = "Delta"
  ) +
  annotate("text", x = -1.1, y = -0.8, label = "Delta = RPDI", 
           angle = 20, size = 2, color = "gray50") +
  theme_q1_journal() +
  theme(legend.position = "none")

# PANEL 2: HOTELLING - Delta vs RPDI
hotelling_data <- data_clean |> filter(Model == "HOTELLING")

p2_hotelling <- ggplot(hotelling_data, aes(x = RPDI, y = Delta)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", 
              color = "gray50", linewidth = 0.6) +
  geom_point(aes(shape = Algorithm, color = Shock_Condition), 
             size = POINT_SIZE, alpha = 0.9, stroke = STROKE_SIZE) +
  scale_color_manual(values = shock_colors) +
  scale_shape_manual(values = algo_shapes) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.2)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.2)) +
  labs(
    title = "Hotelling",
    x = "RPDI",
    y = "Delta"
  ) +
  annotate("text", x = 0.23, y = 0.3, label = "Delta = RPDI", 
           angle = 45, size = 2, color = "gray50") +
  theme_q1_journal() +
  theme(legend.position = "none")

# PANEL 3: LINEAR - Delta vs RPDI
linear_data <- data_clean %>% filter(Model == "LINEAR")

p3_linear <- ggplot(linear_data, aes(x = RPDI, y = Delta)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", 
              color = "gray50", linewidth = 0.6) +
  annotate("rect", xmin = 0.45, xmax = 0.95, ymin = -0.15, ymax = 0.18,
           fill = "#FFF3CD", alpha = 0.5) +
  geom_point(aes(shape = Algorithm, color = Shock_Condition), 
             size = POINT_SIZE, alpha = 0.9, stroke = STROKE_SIZE) +
  scale_color_manual(values = shock_colors) +
  scale_shape_manual(values = algo_shapes) +
  scale_x_continuous(breaks = seq(-0.2, 1, by = 0.2)) +
  scale_y_continuous(breaks = seq(-0.5, 1.5, by = 0.5)) +
  labs(
    title = "Linear",
    x = "RPDI",
    y = "Delta"
  ) +
  annotate("text", x = 0.25, y = 0.38, label = "Delta = RPDI", 
           angle = 25, size = 2, color = "gray50") +
  annotate("text", x = 0.70, y = -0.28, 
           label = "Decoupling", 
           size = 2.5, color = "#B8860B", fontface = "bold") +
  theme_q1_journal() +
  theme(legend.position = "none")

# Ensure Algorithm factor levels match the order in algo_shapes
data_clean$Algorithm <- factor(data_clean$Algorithm, 
                               levels = c("Q-learning", "PSO", "DQN", "DDPG"))
# CREATE SHARED LEGEND
p_legend <- ggplot(data_clean, aes(x = RPDI, y = Delta)) +
  geom_point(aes(shape = Algorithm), size = POINT_SIZE, color = "black") +
  geom_point(aes(color = Shock_Condition), size = 2, shape = 16) +
  scale_color_manual(values = shock_colors, name = "Shock Condition") +
  scale_shape_manual(values = algo_shapes, name = "Algorithm") +
  guides(
    shape = guide_legend(
         order = 1,
         title.position = "top",
         ncol = 1,
         override.aes = list(
                stroke = STROKE_SIZE
         )
    ),
    color = guide_legend(order = 2, title.position = "top", ncol = 1)
  ) +
  theme_q1_journal() +
  theme(
    legend.box = "vertical",
    legend.spacing.y = unit(0.6, "cm"),
    legend.title = element_text(size = 9, face = "bold"),
    legend.text = element_text(size = 8),
    legend.margin = margin(t = 0, r = 5, b = 0, l = 5)
  )

legend_grob <- get_legend(p_legend)

# COMBINE ALL PANELS
combined_plots <- (p1_logit | p2_hotelling | p3_linear) +
  plot_layout(ncol = 3, widths = c(1, 1, 1))

final_figure <- plot_grid(
  combined_plots,
  legend_grob,
  ncol = 2,
  rel_widths = c(1, 0.15),
  align = "h",
  axis = "tb"
)

title_grob <- ggdraw() + 
  draw_label(
    "Relationship between Delta and RPDI",
    fontface = 'bold',
    fontfamily = 'serif',
    size = 12
  )

final_plot <- plot_grid(
  title_grob,
  final_figure,
  ncol = 1,
  rel_heights = c(0.06, 1)
) + theme(plot.margin = margin(5, 10, 5, 5))

# SAVE FIGURES (Q1 Journal Standards)

# PNG - 600 DPI
ggsave(
  filename = "./figures/Figure1_Delta_vs_RPDI_AllMarkets.png",
  plot = final_plot,
  width = 180,
  height = 70,
  units = "mm",
  dpi = 600,
  bg = "white"
)

# PDF - Vector format
ggsave(
  filename = "./figures/Figure1_Delta_vs_RPDI_AllMarkets.pdf",
  plot = final_plot,
  width = 180,
  height = 70,
  units = "mm",
  device = "pdf"
)

cat("\n")
cat("============================================================\n")
cat("  FIGURE 1: Delta vs RPDI - All Market Structures\n")
cat("  Q1 JOURNAL PUBLICATION STANDARD\n")
cat("============================================================\n")
cat("\n")
cat("Files created:\n")
cat("  - Figure1_Delta_vs_RPDI_AllMarkets.png  (600 DPI)\n")
cat("  - Figure1_Delta_vs_RPDI_AllMarkets.pdf  (Vector)\n")
cat("\n")
