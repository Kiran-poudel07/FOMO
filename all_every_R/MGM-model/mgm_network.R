# ========================================================
# Mixed Graphical Model (MGM) for FOMO Dataset (Corrected)
# ========================================================

if (!require("mgm")) install.packages("mgm", dependencies = TRUE)
if (!require("qgraph")) install.packages("qgraph", dependencies = TRUE)
if (!require("psych")) install.packages("psych", dependencies = TRUE)

library(mgm)
library(qgraph)
library(psych)

# Load dataset
df <- read.csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every_R\\all_score_data_sets.csv", header = TRUE)

# Select Q7 to Q36 columns (with original long names)
q_cols <- grep("^Q([7-9]|[1-2][0-9]|3[0-6])", names(df), value = TRUE)

# Rename Q7–Q36 to short names Q7 ... Q36
new_names <- paste0("Q", 7:36)
names(df)[match(q_cols, names(df))] <- new_names

# Define binary variables
binary_vars <- c("heavy_user", "active_user")

# Subset data with Q7-Q36 + heavy_user + active_user only
mgm_data <- df[, c(new_names, binary_vars)]

# Convert Q7–Q36 to numeric
for (col in new_names) {
  mgm_data[[col]] <- as.numeric(as.character(mgm_data[[col]]))
}

# Convert binary vars to factor (categorical)
for (col in binary_vars) {
  mgm_data[[col]] <- as.factor(mgm_data[[col]])
}

# Remove rows with missing values
mgm_data <- na.omit(mgm_data)

# Prepare types vector:
# Q7-Q36 numeric => "g"
# heavy_user, active_user categorical => "c"
types <- c(rep("g", length(new_names)), rep("c", length(binary_vars)))

# Prepare levels vector:
# numeric vars => 1
# binary vars => 2
levels <- c(rep(1, length(new_names)), rep(2, length(binary_vars)))

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

# Prepare color vector for nodes:
# heavy_user, active_user: distinct colors
# Q7–Q11: color1
# Q12–Q16: color2
# Q17–Q21: color3
# Q22–Q26: color4
# Q27–Q31: color5
# Q32–Q36: color6
node_colors <- c(
  "red",    # heavy_user
  "blue",   # active_user
  rep("skyblue", 5),    # Q7–Q11
  rep("lightgreen", 5), # Q12–Q16
  rep("orange", 5),     # Q17–Q21
  rep("gold", 5),       # Q22–Q26
  rep("pink", 5),       # Q27–Q31
  rep("purple", 5)      # Q32–Q36
)

# Plot the network
pdf("fomo_mgm_network_updated.pdf", width = 14, height = 10)
qgraph(mgm_fit$pairwise$wadj,
       layout = "spring",
       labels = colnames(mgm_data),
       color = node_colors,
       edge.color = ifelse(mgm_fit$pairwise$signs == 1, "darkgreen", "red"),
       title = "FOMO MGM Network with heavy_user & active_user")
dev.off()

cat("Updated MGM model complete! Network saved as 'fomo_mgm_network_updated.pdf'\n")
