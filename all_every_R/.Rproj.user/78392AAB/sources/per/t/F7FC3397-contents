# ========================================================
# Psychological Network Analysis (PNA) for FOMO Data - FINAL
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

# 2. Load dataset
cat("Loading dataset...\n")
df <- read.csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every_R\\all_score_data_sets.csv", header = TRUE)

# 3. Extract Q7 to Q36 columns and rename consistently
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

# 5. Create output directory
output_dir <- "PNA_Results"
if(!dir.exists(output_dir)) dir.create(output_dir)

# 6. Estimate network
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

# 7. Plot the network
cat("Creating network plot...\n")
pdf(file.path(output_dir, "fomo_pna_network_plot.pdf"), width = 12, height = 10)
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

# 8. Calculate centrality measures
cat("Calculating centrality measures...\n")
centrality_results <- qgraph::centrality_auto(network_estimate$graph)

# Print centrality
print(centrality_results)

# Export centrality measures
write.csv(centrality_results$node.centrality, file.path(output_dir, "fomo_centrality_measures.csv"))

# 9. Enhanced centrality plot (including Expected Influence)
cat("Creating centrality plot...\n")
pdf(file.path(output_dir, "fomo_pna_centrality_plot.pdf"), width = 12, height = 8)

centrality_measures <- centrality_results$node.centrality
plot_data <- centrality_measures[, c("Strength", "Betweenness", "Closeness", "ExpectedInfluence")]
plot_data <- scale(plot_data) # Standardize for comparison

barplot(t(plot_data), 
        beside = TRUE, 
        col = c("lightblue", "salmon", "lightgreen", "orange"),
        names.arg = rownames(plot_data),
        las = 2,
        cex.names = 0.8,
        main = "Centrality Measures Comparison",
        ylab = "Standardized Score")
legend("topright", 
       legend = c("Strength", "Betweenness", "Closeness", "Expected Influence"),
       fill = c("lightblue", "salmon", "lightgreen", "orange"))

dev.off()

# 10. Stability analysis with bootstrapping
cat("Running bootstrap analysis (this may take some time)...\n")
set.seed(123)
boot_results <- bootnet::bootnet(
  network_estimate,
  nBoots = 1000,  # Increased for better stability
  type = "case"
)

# 11. Plot bootstrap results
cat("Creating bootstrap plots...\n")
pdf(file.path(output_dir, "fomo_pna_bootstrap_plot.pdf"), width = 14, height = 10)
plot(boot_results, labels = TRUE, order = "sample")
dev.off()

# 12. Network Comparison Test (NCT) helper function for pairwise groups
run_nct_pairwise <- function(data, group_var, group_name, output_dir){
  unique_groups <- unique(na.omit(data[[group_var]]))
  n_groups <- length(unique_groups)
  
  if(n_groups < 2){
    cat(sprintf("Not enough groups in %s for NCT.\n", group_name))
    return(NULL)
  }
  
  # For all pairs
  for(i in 1:(n_groups-1)){
    for(j in (i+1):n_groups){
      cat(sprintf("\nRunning NCT for %s groups: %s vs %s\n", group_name, unique_groups[i], unique_groups[j]))
      
      group1_data <- network_data[data[[group_var]] == unique_groups[i], ]
      group2_data <- network_data[data[[group_var]] == unique_groups[j], ]
      
      # Check sample sizes
      if(nrow(group1_data) < 10 || nrow(group2_data) < 10){
        cat("Insufficient sample size for this comparison. Skipping.\n")
        next
      }
      
      nct <- tryCatch({
        NetworkComparisonTest::NCT(
          group1_data,
          group2_data, 
          it = 500,  # Moderate iterations for speed
          test.edges = TRUE
        )
      }, error = function(e){
        cat("Error in NCT:", e$message, "\nSkipping...\n")
        return(NULL)
      })
      
      if(!is.null(nct)){
        print(nct)
        # Save to file
        file_name <- sprintf("NCT_%s_%s_vs_%s.txt", group_name, unique_groups[i], unique_groups[j])
        sink(file.path(output_dir, file_name))
        print(nct)
        sink()
      }
    }
  }
}

# 13. Run NCT for Gender if available (binary)
if("Gender" %in% colnames(df)){
  gender_values <- unique(na.omit(df$Gender))
  if(length(gender_values) == 2){
    cat("\nRunning Network Comparison Test for Gender...\n")
    run_nct_pairwise(df, "Gender", "Gender", output_dir)
  } else {
    cat("\nGender has more than 2 groups, running pairwise NCT...\n")
    run_nct_pairwise(df, "Gender", "Gender", output_dir)
  }
} else {
  cat("\nGender variable not found, skipping gender NCT.\n")
}

# 14. Run NCT for Department if available
if("Department" %in% colnames(df)){
  cat("\nRunning Network Comparison Test for Department (pairwise)...\n")
  run_nct_pairwise(df, "Department", "Department", output_dir)
} else {
  cat("\nDepartment variable not found, skipping Department NCT.\n")
}

# 15. Run NCT for Year if available
if("Year" %in% colnames(df)){
  cat("\nRunning Network Comparison Test for Year (pairwise)...\n")
  run_nct_pairwise(df, "Year", "Year", output_dir)
} else {
  cat("\nYear variable not found, skipping Year NCT.\n")
}

cat("\n=== PNA Analysis Complete! Check folder 'PNA_Results' for outputs. ===\n")
