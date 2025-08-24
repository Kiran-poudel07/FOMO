# ========================================================
# SEM for FOMO with numeric data from Q7 to Q36 (Refined)
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

# 5. Subset the dataframe for SEM
sem_data <- df[, new_names]

# 6. Convert all to numeric (handle factors/characters if any) and remove NA rows
sem_data <- as.data.frame(lapply(sem_data, function(x) as.numeric(as.character(x))))
sem_data <- na.omit(sem_data)

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

# 8. Fit the SEM model
fit <- sem(model, data = sem_data, estimator = "MLR", std.lv = TRUE)

# 9. Print SEM summary with fit measures and standardized estimates
cat("\n===== SEM MODEL SUMMARY =====\n")
summary(fit, fit.measures = TRUE, standardized = TRUE, rsquare = TRUE)

# 10. Print top modification indices > 10
cat("\n===== TOP MODIFICATION INDICES (MI > 10) =====\n")
print(modindices(fit, sort. = TRUE, minimum.value = 10))

# 11. Save SEM path plot as PDF

pdf("fomo_sem_refined_model.pdf", width = 18, height = 12)  # Larger canvas

semPaths(fit,
         what = "std",         # Standardized estimates
         layout = "tree2",     # Improved tree layout
         rotation = 2,         # Better orientation
         sizeMan = 5,          # Size of observed variables
         sizeLat = 7,          # Size of latent variables
         edge.label.cex = 1.2, # Larger edge labels
         nCharNodes = 0,       # Full node names
         title = FALSE,
         whatLabels = "std",   # Show standardized values
         intercepts = FALSE,
         residuals = FALSE,
         optimizeLatRes = TRUE, 
         mar = c(6, 6, 6, 6))  # Margins around plot

dev.off()

