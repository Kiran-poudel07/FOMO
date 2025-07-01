import pandas as pd
from semopy import ModelMeans, semplot, calc_stats

# Step 1: Load the cleaned dataset
df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every\\all_score_data_sets.csv")

# Step 2: Clean Q-column names (extract 'Q7', 'Q8', ..., 'Q36')
df.columns = [col.split('=')[0].strip() if col.startswith('Q') else col for col in df.columns]

# Step 3: Subset only items from Q7 to Q36 (30 questions total)
sem_data = df.loc[:, "Q7":"Q36"]

# Step 4: Clean data: convert to numeric and drop any rows with missing values
sem_data = sem_data.apply(pd.to_numeric, errors='coerce')
sem_data = sem_data.dropna()

# Step 5: Define SEM model with FOMO as second-order latent
model_desc = """
# Measurement model for first-order latent variables
Social_Comparison =~ Q7 + Q8 + Q9 + Q10 + Q11
Peer_Pressure =~ Q12 + Q13 + Q14 + Q15 + Q16
Academic_Burnout =~ Q17 + Q18 + Q19 + Q20 + Q21
Life_Satisfaction =~ Q22 + Q23 + Q24 + Q25 + Q26
Loneliness =~ Q27 + Q28 + Q29 + Q30 + Q31
Perceived_Support =~ Q32 + Q33 + Q34 + Q35 + Q36

# Structural model: FOMO as second-order latent factor
FOMO =~ Social_Comparison + Peer_Pressure + Academic_Burnout + Loneliness + Life_Satisfaction + Perceived_Support
"""

# Step 6: Fit the SEM model
model = ModelMeans(model_desc)
model.fit(sem_data)

# Step 7: Print parameter estimates (loadings, regressions, residuals)
print("\nParameter Estimates:")
print(model.inspect())

# Step 8: Model fit statistics
print("\nModel Fit Stats:")
stats = calc_stats(model)
print(stats)

# Step 9: Save SEM path diagram
print("\nSaving SEM path diagram to 'fomo_sem_model.png'...")
semplot(model, "fomo_sem_model.png")
