# ========================================================
# SEM for FOMO with numeric data from Q7 to Q36 (Refined + Multi-Group Invariance)
# ========================================================

# 1. Load required packages
if (!require("lavaan")) install.packages("lavaan", dependencies = TRUE)
if (!require("semPlot")) install.packages("semPlot", dependencies = TRUE)
if (!require("psych")) install.packages("psych", dependencies = TRUE)

library(lavaan)
library(semPlot)
library(psych)

# 2. Load dataset (with proper headers)
df <- read.csv("all_score_data_sets.csv", header = TRUE)

# 3. Extract Q7 to Q36 columns (names may include question text after '=')
q_cols <- grep("^Q([7-9]|[1-2][0-9]|3[0-6])", names(df), value = TRUE)

# 4. Rename columns to simple Q7, Q8, ..., Q36
new_names <- paste0("Q", 7:36)
if(length(q_cols) == length(new_names)) {
  names(df)[match(q_cols, names(df))] <- new_names
} else {
  stop("Mismatch in number of Q columns found vs expected 30")
}

# 5. Prepare grouping variables as factors
df$Gender <- as.factor(df$Gender)
df$Department <- as.factor(df$Department)
df$Year <- as.factor(df$Year)

# 6. Subset SEM data + grouping vars, remove rows with NA in any relevant column
sem_data_groups <- df[, c(new_names, "Gender", "Department", "Year")]
sem_data_groups <- na.omit(sem_data_groups)

# 7. Define SEM model with MI-based refinements
model <- '
  # First-order latent variables
  Social_Comparison =~ Q7 + Q8 + Q9 + Q10 + Q11
  Peer_Pressure     =~ Q12 + Q13 + Q14 + Q15 + Q16
  Academic_Burnout  =~ Q17 + Q18 + Q19 + Q20 + Q21
  Life_Satisfaction =~ Q22 + Q23 + Q24 + Q25 + Q26
  Loneliness        =~ Q27 + Q28 + Q29 + Q30 + Q31
  Perceived_Support =~ Q32 + Q33 + Q34 + Q35 + Q36

  # Second-order latent variable
  FOMO =~ Social_Comparison + Peer_Pressure + Academic_Burnout +
          Life_Satisfaction + Loneliness + Perceived_Support

  # Residual correlations from Modification Indices
  Academic_Burnout ~~ Life_Satisfaction
  Q12 ~~ Q15
  Q13 ~~ Q14
  Q24 ~~ Q25
  Q18 ~~ Q19
  Q19 ~~ Q21
  Q21 ~~ Q26
  Q22 ~~ Q23
  Q24 ~~ Q26
  Q17 ~~ Q20
  Q23 ~~ Q25
  Q14 ~~ Q35
  Q19 ~~ Q20
  Q27 ~~ Q28
'

# 8. Fit SEM model on full data (no grouping) to get overall fit
fit_full <- sem(model, data = sem_data_groups[, new_names], estimator = "MLR", std.lv = TRUE)

cat("\n===== SEM MODEL SUMMARY (Full Sample) =====\n")
summary(fit_full, fit.measures = TRUE, standardized = TRUE, rsquare = TRUE)

# 9. Save overall SEM path plot as PDF
pdf("fomo_sem_refined_model_overall.pdf", width = 18, height = 12)
semPaths(fit_full,
         what = "std",
         layout = "tree2",
         rotation = 2,
         sizeMan = 5,
         sizeLat = 7,
         edge.label.cex = 1.2,
         nCharNodes = 0,
         title = FALSE,
         whatLabels = "std",
         intercepts = FALSE,
         residuals = FALSE,
         optimizeLatRes = TRUE,
         mar = c(6,6,6,6))
dev.off()

# 10. Function to run measurement invariance tests for a given grouping variable
run_invariance_test <- function(group_var_name, data, model) {
  cat("\n===== Testing invariance for group:", group_var_name, "=====\n")
  
  # Configural invariance (no equality constraints)
  fit_configural <- sem(model, data = data, group = group_var_name, estimator = "MLR", std.lv = TRUE)
  
  # Metric invariance (equal loadings across groups)
  fit_metric <- sem(model, data = data, group = group_var_name, group.equal = c("loadings"), estimator = "MLR", std.lv = TRUE)
  
  # Scalar invariance (equal loadings + intercepts)
  fit_scalar <- sem(model, data = data, group = group_var_name, group.equal = c("loadings", "intercepts"), estimator = "MLR", std.lv = TRUE)
  
  # Compare models
  comparison_metric <- anova(fit_configural, fit_metric)
  comparison_scalar <- anova(fit_metric, fit_scalar)
  
  cat("Configural vs Metric Invariance:\n")
  print(comparison_metric)
  
  cat("\nMetric vs Scalar Invariance:\n")
  print(comparison_scalar)
  
  # Return fits for optional further inspection
  return(list(configural = fit_configural, metric = fit_metric, scalar = fit_scalar))
}

# 11. Run invariance tests for Gender, Department, and Year
fits_gender <- run_invariance_test("Gender", sem_data_groups, model)
fits_department <- run_invariance_test("Department", sem_data_groups, model)
fits_year <- run_invariance_test("Year", sem_data_groups, model)

# 12. Optional: Save group-specific path plots for Gender groups
pdf("fomo_sem_gender_group1.pdf", width = 18, height = 12)
semPaths(fits_gender$configural, what = "std", layout = "tree2", group = 1,
         sizeMan = 5, sizeLat = 7, edge.label.cex = 1.2, title = FALSE, nCharNodes = 0)
dev.off()

pdf("fomo_sem_gender_group2.pdf", width = 18, height = 12)
semPaths(fits_gender$configural, what = "std", layout = "tree2", group = 2,
         sizeMan = 5, sizeLat = 7, edge.label.cex = 1.2, title = FALSE, nCharNodes = 0)
dev.off()

# 13. You can also inspect summaries for each fit if needed, e.g.:
# summary(fits_gender$configural, fit.measures=TRUE, standardized=TRUE)
# summary(fits_department$configural, fit.measures=TRUE, standardized=TRUE)
# summary(fits_year$configural, fit.measures=TRUE, standardized=TRUE)

# END of script
