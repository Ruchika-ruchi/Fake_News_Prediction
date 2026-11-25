import os
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request

# Ensure required NLTK data is available without re-downloading every run
for resource, path in [("stopwords", "corpora/stopwords"), ("punkt_tab", "tokenizers/punkt_tab")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

app = Flask(__name__)

# Exact paths matching Varsha's saved files
MODEL_PATH = os.path.join("model", "logistic_model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    # Prevent AttributeError when loading a scikit-learn 1.9+ model on older versions
    if not hasattr(model, "multi_class"):
        model.multi_class = "auto"
    with open(VECTORIZER_PATH, "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    MODEL_READY = True
    print("✅ Logistic Regression model and TF-IDF vectorizer loaded successfully!")
else:
    model = None
    vectorizer = None
    MODEL_READY = False
    print("⚠️ Warning: Model files not found. Running in Demo Mode.")

stop_words = set(stopwords.words("english"))


def clean_text(text):
    """Exact same NLP preprocessing pipeline used in text_preprocessing.py and predict.py."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news_text", "").strip()

    if not news_text:
        return render_template(
            "index.html",
            error="Please enter a news headline or article before checking."
        )

    cleaned_text = clean_text(news_text)

    if MODEL_READY:
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        label = "Real News" if prediction == 1 else "Fake News"

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vectorized_text)[0]
            confidence = round(max(proba) * 100, 2)
        else:
            confidence = None
    else:
        label = "Fake News (Demo Mode)" if "shocking" in cleaned_text else "Real News (Demo Mode)"
        confidence = 85.0

    return render_template(
        "index.html",
        prediction=label,
        confidence=confidence,
        submitted_text=news_text
    )


if __name__ == "__main__":
    app.run(debug=True)