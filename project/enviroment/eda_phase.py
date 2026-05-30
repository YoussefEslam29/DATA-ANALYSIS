import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Datasets
# Adjust these paths if your filenames differ slightly inside the unzipped folder
train_path = "train_c.csv"  # Based on bean-comp-pytunisia.zip contents
test_path = "test_no_label.csv"

print("--- Loading Data ---")
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print(f"Training shapes: {df_train.shape}")
print(f"Testing shapes: {df_test.shape}\n")

# 2. Inspect Feature Types (Directly answers dataset requirements)
print("--- Dataset Information ---")
print(df_train.info())

# 3. Check for Missing Values (Crucial for report documentation)
missing_vals = df_train.isnull().sum().sum()
print(f"\nTotal Missing Values in Training Set: {missing_vals}")

# 4. Phase 1 Check: Class Distribution Analysis
# This visually answers "Handle class imbalance" from your project prompt
plt.figure(figsize=(10, 5))
sns.countplot(data=df_train, x='y', order=df_train['y'].value_value_counts().index, palette='viridis')
plt.title('Bean Type Class Distribution (Target variable: y)')
plt.xlabel('Bean Type Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('class_distribution.png') # Save this image to drop into your final report!
plt.show()

# Print out the exact text counts for your baseline documentation
print("\n--- Class Counts Summary ---")
print(df_train['y'].value_counts())