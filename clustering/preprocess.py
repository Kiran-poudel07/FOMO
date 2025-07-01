import pandas as pd

# Load the dataset
df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\all_every\\all_score_data_sets.csv")
# print(df.columns.tolist())
#  # Adjust path as needed

# Strip any leading/trailing spaces
df.columns = df.columns.str.strip()

# Step 1: Identify the exact columns for Q7 to Q36 by their full question text
likert_cols = [col for col in df.columns if col.startswith("Q7") or col.startswith("Q8") or col.startswith("Q9") or
               col.startswith("Q10") or col.startswith("Q11") or col.startswith("Q12") or col.startswith("Q13") or
               col.startswith("Q14") or col.startswith("Q15") or col.startswith("Q16") or col.startswith("Q17") or
               col.startswith("Q18") or col.startswith("Q19") or col.startswith("Q20") or col.startswith("Q21") or
               col.startswith("Q22") or col.startswith("Q23") or col.startswith("Q24") or col.startswith("Q25") or
               col.startswith("Q26") or col.startswith("Q27") or col.startswith("Q28") or col.startswith("Q29") or
               col.startswith("Q30") or col.startswith("Q31") or col.startswith("Q32") or col.startswith("Q33") or
               col.startswith("Q34") or col.startswith("Q35") or col.startswith("Q36")]

# Step 2: Create a new DataFrame for those columns
likert_df = df[likert_cols].copy()

# Step 3: Define full column names of positively worded items (life satisfaction + social support)
reverse_items = [
    col for col in likert_cols if any(
        key in col for key in [
            "Q22 = ", "Q23 = ", "Q24 = ", "Q25 = ", "Q26 = ",
            "Q32 = ", "Q33 = ", "Q34 = ", "Q35 = ", "Q36 = "
        ]
    )
]

# Step 4: Reverse-score these columns (6 - value)
for col in reverse_items:
    likert_df[col] = 6 - likert_df[col]

# Step 5: Combine with IDs or other metadata
processed_df = pd.concat([df[["P_ID"]], likert_df], axis=1)

# Save
processed_df.to_csv("clustering/processed_fomo_likert.csv", index=False)
print(" Processed FOMO Likert responses saved!")
