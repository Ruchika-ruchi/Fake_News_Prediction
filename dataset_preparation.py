import pandas as pd

# Load datasets
fake = pd.read_csv("Dataset/Fake.csv")
true = pd.read_csv("Dataset/True.csv")

# Display first 5 records
print("FAKE NEWS DATA:")
print(fake.head())

print("\nREAL NEWS DATA:")
print(true.head())

# Display dataset size
print("\nFake news records:", len(fake))
print("Real news records:", len(true))

# Display column names
print("\nFake columns:", fake.columns.tolist())
print("Real columns:", true.columns.tolist())
# Check missing values

print("\nMissing values in Fake News:")
print(fake.isnull().sum())

print("\nMissing values in Real News:")
print(true.isnull().sum())


# Check duplicate records

print("\nDuplicate records in Fake News:")
print(fake.duplicated().sum())

print("\nDuplicate records in Real News:")
print(true.duplicated().sum())


# Check dataset information

print("\nFake News Information:")
print(fake.info())

print("\nReal News Information:")
print(true.info())
# Remove duplicate records

fake = fake.drop_duplicates(subset=['title', 'text'])
true = true.drop_duplicates(subset=['title', 'text'])

print("\nAfter removing duplicates:")
print("Fake news:", len(fake))
print("Real news:", len(true))
# Assign labels

fake['label'] = 0
true['label'] = 1

print("\nLabels assigned successfully!")
# Combine fake and real news

news = pd.concat([fake, true], ignore_index=True)

print("\nCombined dataset:")
print(news.head())

print("\nTotal records:", len(news))
# Shuffle the dataset

news = news.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDataset shuffled successfully!")
# Save cleaned dataset

news.to_csv("Dataset/news_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")
# Dataset analysis

print("\nFake and Real News Count:")
print(news['label'].value_counts())

print("\nDataset Shape:")
print(news.shape)

print("\nSample Fake News:")
print(news[news['label'] == 0][['title', 'label']].head())

print("\nSample Real News:")
print(news[news['label'] == 1][['title', 'label']].head())