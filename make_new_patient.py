import pandas as pd

df = pd.read_csv("parkinsons.csv")
feature_names = [c for c in df.columns if c not in ["name", "status"]]

# pick any row index you want (this just makes a realistic "new patient")
row_idx = 125

df.loc[[row_idx], feature_names].to_csv("new_patient.csv", index=False)
print("Wrote new_patient.csv with 1 row")
