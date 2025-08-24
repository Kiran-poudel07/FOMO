# ========================================================
# Psychological Network Analysis (PNA) on FOMO Variables
# ========================================================

# 1. Install/load packages
options(repos = c(CRAN = "https://cloud.r-project.org"))

packages <- c("bootnet", "qgraph", "psych")
for(p in packages){
  if (!require(p, character.only = TRUE)) {
    install.packages(p, dependencies = TRUE)
    library(p, character.only = TRUE)
  }
}

# 2. Load dataset
cat("Loading dataset...\n")
df <- read.csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every_R\\all_score_data_sets.csv", header = TRUE)

# 3. Rename Q7 to Q36 for consistency
q_cols <- grep("^Q[7-9]|^Q1[0-9]|^Q2[0-9]|^Q3[0-6]", names(df), value = TRUE)
new_names <- paste0("Q", 7:36)
if(length(q_cols) == length(new_names)) {
  names(df)[match(q_cols, names(df))] <- new_names
} else {
  stop("Column rename mismatch: Q7 to Q36 not found or incomplete.")
}

# 4. Define constructs
constructs <- list(
  Social_Comparison = paste0("Q", 7:11),
  Peer_Pressure = paste0("Q", 12:16),
  Academic_Burnout = paste0("Q", 17:21),
  Life_Satisfaction = paste0("Q", 22:26),
  Loneliness = paste0("Q", 27:31),
  Perceived_Support = paste0("Q", 32:36)
)

# 5. Compute mean scores for each construct
for (construct in names(constructs)) {
  df[[construct]] <- rowMeans(df[, constructs[[construct]]], na.rm = TRUE)
}

# 6. Include heavy_user and active_user, convert to numeric
df$heavy_user <- as.numeric(as.character(df$heavy_user))
df$active_user <- as.numeric(as.character(df$active_user))

# 7. Prepare data for network
network_vars <- c(names(constructs), "heavy_user", "active_user")
network_data <- df[, network_vars]
network_data <- na.omit(network_data)

# 8. Estimate network
cat("Estimating network...\n")
network_estimate <- bootnet::estimateNetwork(
  network_data,
  default = "EBICglasso",
  corMethod = "cor_auto",
  tuning = 0.5
)

# 9. Plot directory
output_dir <- "PNA_Results"
if(!dir.exists(output_dir)) dir.create(output_dir)

# 10. Network plot with node colors
cat("Creating network plot...\n")
node_colors <- c(
  "#FFD700",  # Social_Comparison
  "#FF69B4",  # Peer_Pressure
  "#87CEEB",  # Academic_Burnout
  "#98FB98",  # Life_Satisfaction
  "#DA70D6",  # Loneliness
  "#FFA07A",  # Perceived_Support
  "#B22222",  # heavy_user
  "#008000"   # active_user
)

pdf(file.path(output_dir, "pna_network_plot.pdf"), width = 12, height = 10)
qgraph(
  network_estimate$graph,
  layout = "spring",
  labels = colnames(network_data),
  color = node_colors,
  label.cex = 1.2,
  vsize = 6,
  edge.color = ifelse(network_estimate$graph > 0, "darkgreen", "red"),
  title = "FOMO Psychological Network (Variable Level)"
)
dev.off()

# 11. Centrality plot
cat("Creating centrality plot...\n")
centrality <- qgraph::centrality_auto(network_estimate$graph)
centrality_measures <- centrality$node.centrality

pdf(file.path(output_dir, "centrality_plot.pdf"), width = 12, height = 8)
barplot(t(scale(centrality_measures[, c("Strength", "Betweenness", "Closeness", "ExpectedInfluence")])),
        beside = TRUE,
        col = c("skyblue", "salmon", "lightgreen", "orange"),
        names.arg = rownames(centrality_measures),
        las = 2,
        cex.names = 0.8,
        main = "Centrality Measures (Standardized)",
        ylab = "Z-Score")
legend("topright", legend = colnames(centrality_measures), fill = c("skyblue", "salmon", "lightgreen", "orange"))
dev.off()

# 12. Bootstrapping
cat("Running bootstrapping (takes time)...\n")
set.seed(123)
boot_results <- bootnet::bootnet(
  network_estimate,
  nBoots = 1000,
  type = "case"
)

cat("Saving bootstrap plot...\n")
pdf(file.path(output_dir, "bootstrap_plot.pdf"), width = 14, height = 10)
plot(boot_results, labels = TRUE, order = "sample")
dev.off()

cat("\nAll plots complete. Check the 'PNA_Results' folder.\n")
