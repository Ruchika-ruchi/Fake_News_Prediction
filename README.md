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
Fake_News_Prediction/
├── data/                        # Raw and cleaned news datasets
├── model/
│   ├── model.pkl                # Trained ML classifier (added after training)
│   └── vectorizer.pkl           # Fitted TF-IDF vectorizer
├── static/
│   └── style.css                # Frontend styling
├── templates/
│   └── index.html               # Web UI template
├── dataset_preparation.py       # Dataset loading, merging, and label encoding
├── text_preprocessing.py        # NLP text cleaning and normalization pipeline
├── feature_extraction.py        # TF-IDF vectorization and feature saving
├── app.py                       # Flask backend and /predict routes
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation