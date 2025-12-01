# Load required libraries
library(tidyverse)
library(ggplot2)
library(patchwork)
library(scales)
library(cowplot)

POINT_SIZE <- 1.7
STROKE_SIZE <- 0.8

# Set seed for reproducibility
set.seed(42)

# DATA LOADING AND PREPARATION
data <- read.csv("all_tables.csv", stringsAsFactors = FALSE)

data_long <- data %>%
  pivot_longer(
    cols = c(Agent1_Delta, Agent2_Delta, Agent1_RPDI, Agent2_RPDI, 
             Agent1_Avg_Prices, Agent2_Avg_Prices),
    names_to = c("Agent", ".value"),
    names_pattern = "(Agent[12])_(.*)"
  ) %>%
  rename(Delta = Delta, RPDI = RPDI, Avg_Prices = Avg_Prices)

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

# Remove extreme outliers
data_clean <- data_long %>%
  filter(Delta > -8 & Delta < 3)

# COLOR PALETTE (Colorblind-friendly)
algo_colors <- c(
  "Q-learning" = "#1B9E77",
  "PSO"        = "#7570B3", 
  "DDQN"        = "#D95F02",
  "DDPG"       = "#E7298A"
)

algo_shapes <- c(
  "Q-learning" = 21,
  "PSO"        = 24,
  "DDQN"        = 22,
  "DDPG"       = 23
)

algo_linetypes <- c(
  "Q-learning" = "solid",
  "PSO"        = "solid",
  "DDQN"        = "solid",
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
        size = rel(0.9),
        hjust = 0.5,
        margin = margin(b = 6),
        color = "gray40"
      ),
      axis.title = element_text(size = rel(1.0), face = "plain"),
      axis.title.x = element_text(margin = margin(t = 8)),
      axis.title.y = element_text(margin = margin(r = 8)),
      axis.text = element_text(size = rel(0.9), color = "black"),
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      panel.grid.major = element_line(color = "gray88", linewidth = 0.35),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.6),
      legend.title = element_text(size = rel(0.95), face = "bold"),
      legend.text = element_text(size = rel(0.85)),
      legend.key.size = unit(1.0, "lines"),
      legend.key.width = unit(1.5, "lines"),
      legend.background = element_rect(fill = "white", color = NA),
      legend.key = element_rect(fill = "white", color = NA),
      plot.margin = margin(t = 8, r = 8, b = 8, l = 8)
    )
}

# PREPARE AGGREGATED DATA
shock_data <- data_clean %>%
  group_by(Model, Algorithm, Shock_Condition) %>%
  summarise(
    Mean_Delta = mean(Delta, na.rm = TRUE),
    SE_Delta = sd(Delta, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

# PANEL 1: LOGIT - Shock Impact
logit_shock <- shock_data %>% filter(Model == "LOGIT")

p1_logit <- ggplot(logit_shock, 
                   aes(x = Shock_Condition, y = Mean_Delta, 
                       color = Algorithm, group = Algorithm)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0,
           fill = "#FFEEEE", alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40", linewidth = 0.5) +
  geom_hline(yintercept = 1, linetype = "dotted", color = "gray40", linewidth = 0.5) +
  # geom_line(aes(linetype = Algorithm), linewidth = 0.9) +
  geom_point(aes(shape = Algorithm), size = POINT_SIZE, stroke = STROKE_SIZE) +
  scale_color_manual(values = algo_colors) +
  scale_shape_manual(values = algo_shapes) +
  scale_linetype_manual(values = algo_linetypes) +
  scale_y_continuous(
    limits = c(-7, 1.5),
    breaks = seq(-6, 1, by = 1),
    minor_breaks = NULL
  ) +
  labs(
    title = "Logit",
    # subtitle = "Collapse under shocks",
    x = NULL,
    y = "Delta"
  ) +
  # annotate("text", x = 0.55, y = 0.25, label = "Delta = 0", 
  #          size = 2.3, color = "gray40", hjust = 0) +
  # annotate("text", x = 0.55, y = 1.25, label = "Delta = 1", 
  #          size = 2.3, color = "gray40", hjust = 0) +
  theme_q1_journal() +
  theme(
    legend.position = "none",
    plot.subtitle = element_text(color = "#D55E00", face = "italic"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8)
  )

# PANEL 2: HOTELLING - Shock Impact
hotelling_shock <- shock_data %>% filter(Model == "HOTELLING")

p2_hotelling <- ggplot(hotelling_shock, 
                       aes(x = Shock_Condition, y = Mean_Delta, 
                           color = Algorithm, group = Algorithm)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0,
           fill = "#FFEEEE", alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40", linewidth = 0.5) +
  geom_hline(yintercept = 1, linetype = "dotted", color = "gray40", linewidth = 0.5) +
  #geom_line(aes(linetype = Algorithm), linewidth = 0.9) +
  geom_point(aes(shape = Algorithm), size = POINT_SIZE, stroke = STROKE_SIZE) +
  scale_color_manual(values = algo_colors) +
  scale_shape_manual(values = algo_shapes) +
  scale_linetype_manual(values = algo_linetypes) +
  scale_y_continuous(
    limits = c(-0.3, 1.2),
    breaks = seq(0, 1, by = 0.25),
    minor_breaks = NULL
  ) +
  labs(
    title = "Hotelling",
    # subtitle = "Stable across shocks",
    x = NULL,
    y = "Delta"
  ) +
  # annotate("text", x = 0.55, y = 0.08, label = "Delta = 0", 
  #          size = 2.3, color = "gray40", hjust = 0) +
  # annotate("text", x = 0.55, y = 1.08, label = "Delta = 1", 
  #          size = 2.3, color = "gray40", hjust = 0) +
  theme_q1_journal() +
  theme(
    legend.position = "none",
    plot.subtitle = element_text(color = "#009E73", face = "italic"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8)
  )

# PANEL 3: LINEAR - Shock Impact
linear_shock <- shock_data %>% filter(Model == "LINEAR")

p3_linear <- ggplot(linear_shock, 
                    aes(x = Shock_Condition, y = Mean_Delta, 
                        color = Algorithm, group = Algorithm)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0,
           fill = "#FFEEEE", alpha = 0.6) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = 1, ymax = Inf,
           fill = "#FFFACD", alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40", linewidth = 0.5) +
  geom_hline(yintercept = 1, linetype = "dotted", color = "gray40", linewidth = 0.5) +
  # geom_line(aes(linetype = Algorithm), linewidth = 0.9) +
  geom_point(aes(shape = Algorithm), size = POINT_SIZE, stroke = STROKE_SIZE) +
  scale_color_manual(values = algo_colors) +
  scale_shape_manual(values = algo_shapes) +
  scale_linetype_manual(values = algo_linetypes) +
  scale_y_continuous(
    limits = c(-0.3, 1.6),
    breaks = seq(0, 1.5, by = 0.5),
    minor_breaks = NULL
  ) +
  labs(
    title = "Linear",
    # subtitle = "Supra-monopoly profits",
    x = NULL,
    y = "Delta"
  ) +
  # annotate("text", x = 0.55, y = 0.08, label = "Delta = 0", 
  #          size = 2.3, color = "gray40", hjust = 0) +
  # annotate("text", x = 0.55, y = 1.08, label = "Delta = 1", 
  #          size = 2.3, color = "gray40", hjust = 0) +
  theme_q1_journal() +
  theme(
    legend.position = "none",
    plot.subtitle = element_text(color = "#CC79A7", face = "italic"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8)
  )

# CREATE SHARED LEGEND
p_legend <- ggplot(shock_data, aes(x = Shock_Condition, y = Mean_Delta)) +
  geom_point(aes(color = Algorithm, shape = Algorithm), size = POINT_SIZE) +
  scale_color_manual(
    values = algo_colors,
    name = "Algorithm",
    breaks = c("Q-learning", "PSO", "DDQN", "DDPG")  # Add this to all scales
  ) +
  scale_shape_manual(
    values = algo_shapes,
    name = "Algorithm",
    breaks = c("Q-learning", "PSO", "DDQN", "DDPG")  # Add this to all scales
  ) +
  scale_linetype_manual(
    values = algo_linetypes,
    name = "Algorithm",
    breaks = c("Q-learning", "PSO", "DDQN", "DDPG")  # Add this to all scales
  ) +
  guides(
    color = guide_legend(
         title.position = "top",
         ncol = 1,
         order = 1,
         override.aes = list(
                stroke = STROKE_SIZE,
                color = algo_colors
         )
    ),
    shape = guide_legend(title.position = "top", ncol = 1, order = 1),
    linetype = guide_legend(title.position = "top", ncol = 1, order = 1)
  ) +
  theme_q1_journal() +
  theme(
    legend.title = element_text(size = 9, face = "bold"),
    legend.text = element_text(size = 8),
    legend.margin = margin(t = 0, r = 0, b = 0, l = 0)
  )

legend_grob <- get_legend(p_legend)

# COMBINE ALL PANELS
x_label <- ggdraw() +
  draw_label("Shock Condition", size = 10, fontfamily = 'serif')

combined_plots <- (p1_logit | p2_hotelling | p3_linear) +
  plot_layout(ncol = 3, widths = c(1, 1, 1))

plots_with_xlabel <- plot_grid(
  combined_plots,
  x_label,
  ncol = 1,
  rel_heights = c(1, 0.05)
)

final_figure <- plot_grid(
  plots_with_xlabel,
  legend_grob,
  ncol = 2,
  rel_widths = c(1, 0.12),
  align = "h",
  axis = "tb"
)

title_grob <- ggdraw() + 
  draw_label(
    "Shock Impact on Delta Across Market Structures",
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

# SAVE FIGURES
ggsave(
  filename = "./figures/Figure2_Shock_Impact_AllMarkets.png",
  plot = final_plot,
  width = 200,
  height = 100,
  units = "mm",
  dpi = 600,
  bg = "white"
)

ggsave(
  filename = "./figures/Figure2_Shock_Impact_AllMarkets.pdf",
  plot = final_plot,
  width = 200,
  height = 100,
  units = "mm",
  device = "pdf"
)

cat("\n")
cat("============================================================\n")
cat("  FIGURE 2: Shock Impact Across Market Structures\n")
cat("============================================================\n")
cat("\n")
cat("Files created:\n")
cat("  - Figure2_Shock_Impact_AllMarkets.png  (600 DPI)\n")
cat("  - Figure2_Shock_Impact_AllMarkets.pdf  (Vector)\n")
# cat("  - Figure2_Shock_Impact_AllMarkets.tiff (600 DPI, LZW)\n")
cat("\n")
cat("Key patterns visualized:\n")
cat("  - Logit: Catastrophic collapse under shocks A & C\n")
cat("  - Hotelling: Remarkable stability (nearly flat lines)\n")
cat("  - Linear: Profit inflation above monopoly level\n")
cat("\n")
