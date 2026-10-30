# app/main.py
# Streamlit app: classifies an SMS with a machine learning model (SVM)
# and, if available, a deep learning model (neural network).

import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

# Main project folder (the folder above "app")
FOLDER = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="SMS Spam Classifier", layout="centered")


@st.cache_resource
def load_ml():
    vectorizer = joblib.load(FOLDER / "tfidf_vectorizer.pkl")
    model = joblib.load(FOLDER / "svm_model.pkl")
    return vectorizer, model


@st.cache_resource
def load_dl():
    """Load the neural network. Returns None if it is not available."""
    path = FOLDER / "dl_model.keras"
    if not path.exists():
        return None
    try:
        from tensorflow import keras

        return keras.models.load_model(path)
    except Exception:
        return None


def load_results():
    results = []
    for name in ("results_ml.json", "results_dl.json"):
        path = FOLDER / name
        if path.exists():
            with open(path) as f:
                results.append(json.load(f))
    return results


vectorizer, svm_model = load_ml()
dl_model = load_dl()

st.title("📩 SMS Spam Detection")
st.markdown(
    "Machine learning (SVM) **vs** deep learning (neural network). "
    "Enter a message below to check if it is spam or not."
)

message = st.text_area("✍️ Enter SMS text here:")

if st.button("🔍 Predict"):
    if message.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        col1, col2 = st.columns(2)

        # Machine learning model
        with col1:
            st.subheader("Machine learning")
            st.caption("SVM + TF-IDF")
            label = svm_model.predict(vectorizer.transform([message]))[0]
            if str(label).lower() in ("spam", "1"):
                st.error("🚨 SPAM")
            else:
                st.success("✅ NOT SPAM")

        # Deep learning model
        with col2:
            st.subheader("Deep learning")
            st.caption("Embedding + BiLSTM")
            if dl_model is None:
                st.info("Deep learning model not available.")
            else:
                probability = float(dl_model.predict(np.array([[message]]), verbose=0)[0][0])
                if probability >= 0.5:
                    st.error(f"🚨 SPAM ({probability:.1%})")
                else:
                    st.success(f"✅ NOT SPAM ({1 - probability:.1%})")

results = load_results()
if results:
    st.divider()
    st.subheader("Model comparison (test set)")
    st.table(
        {
            "Model": [r["model"] for r in results],
            "Accuracy": [f"{r['accuracy']:.4f}" for r in results],
            "F1-score (spam)": [f"{r['f1_spam']:.4f}" for r in results],
            "Training time (s)": [r["training_seconds"] for r in results],
        }
    )
