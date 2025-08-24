# ========================================================
# Psychological Network Analysis (PNA) for FOMO Data - WORKING VERSION
# ========================================================

# 1. Set CRAN mirror and install/load necessary packages
options(repos = c(CRAN = "https://cloud.r-project.org"))

packages <- c("bootnet", "qgraph", "NetworkComparisonTest", "psych")
for(p in packages){
  if (!require(p, character.only = TRUE)) {
    install.packages(p, dependencies = TRUE)
    library(p, character.only = TRUE)
  }
}

# 2. Load your dataset
cat("Loading dataset...\n")
df <- read.csv("all_score_data_sets.csv", header = TRUE)

# 3. Extract Q7 to Q36 columns
q_cols <- grep("^Q([7-9]|[1-2][0-9]|3[0-6])", names(df), value = TRUE)
new_names <- paste0("Q", 7:36)
if(length(q_cols) == length(new_names)) {
  names(df)[match(q_cols, names(df))] <- new_names
} else {
  stop("Mismatch in number of Q columns")
}

# 4. Prepare network data
cat("Preparing network data...\n")
network_data <- df[, new_names]
network_data <- as.data.frame(lapply(network_data, function(x) as.numeric(as.character(x))))
network_data <- na.omit(network_data)

# 5. Estimate network
cat("Estimating network...\n")
network_estimate <- bootnet::estimateNetwork(
  network_data,
  default = "EBICglasso",
  corMethod = "cor_auto",
  tuning = 0.5
)

# Calculate network density
network_density <- mean(network_estimate$graph != 0)
cat(sprintf("\nNetwork density: %.2f%%\n", network_density*100))

# 6. Plot the network
cat("Creating network plot...\n")
pdf("fomo_pna_network_plot.pdf", width = 12, height = 10)
qgraph::qgraph(
  network_estimate$graph,
  layout = "spring",
  labels = colnames(network_data),
  label.cex = 1.2,
  vsize = 6,
  color = "lightblue",
  edge.color = ifelse(network_estimate$graph > 0, "darkgreen", "red"),
  title = "FOMO Network Analysis"
)
dev.off()

# 7. Calculate centrality measures
cat("Calculating centrality measures...\n")
centrality_results <- qgraph::centrality_auto(network_estimate$graph)
print(centrality_results)

# Export centrality measures
write.csv(centrality_results$node.centrality, "fomo_centrality_measures.csv")

# 8. Plot centrality indices - USING QGRAPH ALTERNATIVE
cat("Creating centrality plot...\n")
pdf("fomo_pna_centrality_plot.pdf", width = 12, height = 8)

# Create a custom centrality plot using qgraph
centrality_measures <- centrality_results$node.centrality
plot_data <- centrality_measures[, c("Strength", "Betweenness", "Closeness")]
plot_data <- scale(plot_data) # Standardize for comparison

barplot(t(plot_data), 
        beside = TRUE, 
        col = c("lightblue", "salmon", "lightgreen"),
        names.arg = rownames(plot_data),
        las = 2,
        cex.names = 0.8,
        main = "Centrality Measures Comparison",
        ylab = "Standardized Score")
legend("topright", 
       legend = c("Strength", "Betweenness", "Closeness"),
       fill = c("lightblue", "salmon", "lightgreen"))

dev.off()

# 9. Stability analysis
cat("Running bootstrap analysis...\n")
set.seed(123)
boot_results <- bootnet::bootnet(
  network_estimate,
  nBoots = 500,  # Reduced for speed, increase for final analysis
  type = "case"
)

# 10. Plot bootstrap results
cat("Creating bootstrap plots...\n")
pdf("fomo_pna_bootstrap_plot.pdf", width = 14, height = 10)
plot(boot_results, labels = TRUE, order = "sample")
dev.off()

# 11. OPTIONAL: Network comparison test
if("Gender" %in% colnames(df)){
  gender_values <- unique(na.omit(df$Gender))
  if(length(gender_values) == 2){
    cat("\nRunning Network Comparison Test...\n")
    
    group1_data <- network_data[df$Gender == gender_values[1] & !is.na(df$Gender), ]
    group2_data <- network_data[df$Gender == gender_values[2] & !is.na(df$Gender), ]
    
    if(nrow(group1_data) > 10 && nrow(group2_data) > 10){
      nct <- NetworkComparisonTest::NCT(
        group1_data,
        group2_data, 
        it = 500,  # Reduced for speed
        test.edges = TRUE
      )
      print(nct)
      sink("nct_results.txt")
      print(nct)
      sink()
    } else {
      cat("Insufficient sample size. Skipping NCT.\n")
    }
  }
}

cat("\nAnalysis complete! Check your working directory for outputs.\n")