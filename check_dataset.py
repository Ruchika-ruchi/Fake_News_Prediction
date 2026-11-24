import pandas as pd

# Load Fake News dataset
fake = pd.read_csv(
    r"C:\Users\Varsha P\OneDrive\Documents\Fake.csv"
)

# Load True News dataset
true = pd.read_csv(
    r"C:\Users\Varsha P\OneDrive\Documents\cleaned_true.csv"
)

print("FAKE NEWS DATASET")
print("Shape:", fake.shape)
print(fake.head())

print("\nTRUE NEWS DATASET")
print("Shape:", true.shape)
print(true.head())
print("\nFAKE DATASET COLUMNS:")
print(fake.columns.tolist())

print("\nTRUE DATASET COLUMNS:")
print(true.columns.tolist())
print("\nFAKE COLUMNS:")
print(fake.columns.tolist())

print("\nTRUE COLUMNS:")
print(true.columns.tolist())
print("\nTRUE DATASET LABEL VALUES:")
print(true['label'].value_counts(dropna=False))
# Assign labels
fake['label'] = 0
true['label'] = 1

# Combine both datasets
import pandas as pd

combined = pd.concat([fake, true], ignore_index=True)

# Save combined dataset
combined.to_csv('combined_news.csv', index=False)

print("\nCOMBINED DATASET")
print("Shape:", combined.shape)
print(combined['label'].value_counts())
print("\nMISSING VALUES IN COMBINED DATASET:")
print(combined.isnull().sum())

print("\nDUPLICATE ROWS:")
print(combined.duplicated().sum())
# Remove duplicate records
combined = combined.drop_duplicates()

# Check updated dataset
print("\nDATASET AFTER REMOVING DUPLICATES")
print("Shape:", combined.shape)

print("\nLABEL DISTRIBUTION:")
print(combined['label'].value_counts())

# Save cleaned dataset
combined.to_csv('cleaned_dataset.csv', index=False)

print("\nCleaned dataset saved successfully!")