# ==========================================================
# MGM Model with heavy_user, active_user + 6 numeric constructs
# ==========================================================

library(mgm)
library(qgraph)
library(psych)

# Load dataset
file_path <- "C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every_R\\all_score_data_sets.csv"
df <- read.csv(file_path, header = TRUE)

# Variables to include:
# Binary: heavy_user, active_user (coded 0/1)
# Numeric constructs:
construct_vars <- c(
  "social_comparison",
  "peer_pressure",
  "academic_burnout",
  "life_satisfaction",
  "loneliness",
  "perceived_support"
)
binary_vars <- c("heavy_user", "active_user")

all_vars <- c(binary_vars, construct_vars)

# Subset data
mgm_data <- df[, all_vars]

# Convert numeric constructs to numeric
for (col in construct_vars) {
  mgm_data[[col]] <- as.numeric(as.character(mgm_data[[col]]))
}

# Convert binary vars to factor (categorical)
for (col in binary_vars) {
  mgm_data[[col]] <- as.factor(mgm_data[[col]])
}

# Remove rows with missing data
mgm_data <- na.omit(mgm_data)

# Types vector: "c" for categorical(binary), "g" for Gaussian(numeric)
types <- c(rep("c", length(binary_vars)), rep("g", length(construct_vars)))

# Levels vector: 2 levels for binary vars, 1 for numeric constructs
levels <- c(rep(2, length(binary_vars)), rep(1, length(construct_vars)))

# Colors: one unique color per node (order must match all_vars)
node_colors <- c(
  "#FF4500",  # heavy_user (orange-red)
  "#1E90FF",  # active_user (dodger blue)
  "#FFB6C1",  # social_comparison (light pink)
  "#98FB98",  # peer_pressure (pale green)
  "#FFD700",  # academic_burnout (gold)
  "#FFA07A",  # life_satisfaction (light salmon)
  "#DA70D6",  # loneliness (orchid)
  "#40E0D0"   # perceived_support (turquoise)
)

# Fit MGM model
set.seed(123)
mgm_fit <- mgm(
  data = data.matrix(mgm_data),
  type = types,
  level = levels,
  lambdaSel = "EBIC",
  lambdaGam = 0.25,
  k = 2
)

# Plot network
pdf("fomo_mgm_heavy_active_plus_constructs_network.pdf", width = 14, height = 10)
qgraph(mgm_fit$pairwise$wadj,
       layout = "spring",
       labels = colnames(mgm_data),
       color = node_colors,
       edge.color = ifelse(mgm_fit$pairwise$signs == 1, "darkgreen", "red"),
       title = "FOMO MGM Network: Heavy & Active Users + Constructs")
dev.off()

cat("GM network with heavy_user, active_user + constructs saved as 'fomo_mgm_heavy_active_plus_constructs_network.pdf'\n")
