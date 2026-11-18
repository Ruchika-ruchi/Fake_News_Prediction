
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

# Step 1: Load preprocessed dataset
news = pd.read_csv("Dataset/preprocessed_news.csv")

print("Dataset loaded successfully!")
print("Original records:", len(news))

# Step 2: Remove missing and empty records
news = news.dropna(subset=['cleaned_text', 'label'])

news['cleaned_text'] = news['cleaned_text'].astype(str)

news = news[news['cleaned_text'].str.strip() != '']

print("Records after cleaning:", len(news))

# Step 3: Separate text and labels
X = news['cleaned_text']
y = news['label']

# Step 4: Split dataset into training and testing
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDataset split completed!")
print("Training records:", len(X_train_text))
print("Testing records:", len(X_test_text))

# Step 5: Apply TF-IDF
tfidf = TfidfVectorizer(max_features=5000)

X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)

print("\nTF-IDF Feature Extraction Completed!")
print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

# Step 6: Save TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(tfidf, file)

# Step 7: Save training and testing data
training_data = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test
}

with open("training_data.pkl", "wb") as file:
    pickle.dump(training_data, file)

print("\nTraining and testing data saved successfully!")
print("TF-IDF vectorizer saved successfully!")
print("\nFEATURE EXTRACTION COMPLETED!")
