import os
import joblib
from flask import Flask, render_template, request

try:
    from utils import preprocess_text
except ImportError:
    def preprocess_text(text):
        return text.lower().strip()

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    MODEL_READY = True
else:
    model = None
    vectorizer = None
    MODEL_READY = False


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

    cleaned_text = preprocess_text(news_text)

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