import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load trained model
with open("model/logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load vectorizer
with open("model/vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)

# Same preprocessing used during training
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

print("Fake News Detection System")

news = input("\nEnter news text: ")

cleaned_news = clean_text(news)
features = tfidf.transform([cleaned_news])

prediction = model.predict(features)

if prediction[0] == 1:
    print("\nPrediction: REAL NEWS")
else:
    print("\nPrediction: FAKE NEWS")