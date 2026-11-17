# 📰 Fake News Detection System

A machine learning and natural language processing (NLP) web application built with **Python** and **Flask** that classifies news headlines or full articles as **Real** or **Fake**.

---

## 🚀 Features
- **Real-Time Classification:** Paste any news headline or full article to check its authenticity.
- **NLP Text Preprocessing:** Automated text cleaning including lowercasing, punctuation/URL removal, stopword filtering, and lemmatization.
- **TF-IDF Vectorization & ML:** Uses TF-IDF features paired with supervised ML classifiers (Logistic Regression / Passive Aggressive Classifier) to output predictions and confidence scores.
- **Clean Web Interface:** Simple, responsive Flask UI for quick testing and live demos.

---

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Data & NLP:** `pandas`, `numpy`, `NLTK`
- **Machine Learning:** `scikit-learn`, `joblib`
- **Backend:** `Flask`
- **Frontend:** `HTML5`, `CSS3`

---

## 📂 Project Structure
```text
├── data/
│   └── cleaned_data.csv        # Preprocessed dataset
├── model/
│   ├── model.pkl               # Trained ML classifier
│   └── vectorizer.pkl          # Fitted TF-IDF vectorizer
├── static/
│   └── style.css               # Frontend styling
├── templates/
│   └── index.html              # Web UI template
├── utils.py                    # Shared NLP text preprocessing pipeline
├── train.py                    # Model training & evaluation script
├── app.py                      # Flask backend & prediction routes
├── requirements.txt            # Python dependencies
└── README.md
