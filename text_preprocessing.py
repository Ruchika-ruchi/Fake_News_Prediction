
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load cleaned dataset
news = pd.read_csv("Dataset/news_dataset.csv")

print("Dataset loaded successfully!")
print("Total records:", len(news))

# Select English stopwords
stop_words = set(stopwords.words('english'))

# Function to clean news text
def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize text into words
    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Join words back into a sentence
    return ' '.join(words)


# Apply preprocessing to news text
news['cleaned_text'] = news['text'].apply(clean_text)

# Display results
print("\nOriginal Text:")
print(news['text'].iloc[0])

print("\nCleaned Text:")
print(news['cleaned_text'].iloc[0])

# Save preprocessed dataset
news.to_csv("Dataset/preprocessed_news.csv", index=False)

print("\nPreprocessed dataset saved successfully!")
