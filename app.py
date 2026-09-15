from pathlib import Path
import re
import unicodedata
import joblib
import streamlit as st

# 1. Configuration de l'interface Streamlit
st.set_page_config(page_title="SMS Spam Classifier", layout="centered")

# 2. Chargement du modèle et du vectoriseur
# Repère les fichiers dans le même dossier que app.py
BASE_DIR = Path(__file__).resolve().parent
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = BASE_DIR / "svm_model.pkl"

@st.cache_resource
def load_resources():
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model

try:
    vectorizer, model = load_resources()
except Exception as e:
    st.error(f"Erreur de chargement des fichiers `.pkl` : {e}")

# 3. Fonction de nettoyage robuste & multilingue
def advanced_preprocess(text):
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower()
    
    leet_dict = {'@': 'a', '0': 'o', '1': 'i', '!': 'i', '$': 's', '3': 'e'}
    for char, replacement in leet_dict.items():
        text = text.replace(char, replacement)
        
    text = re.sub(r'http\S+|www\.\S+', ' tokenurl ', text)
    text = re.sub(r'\b\d{5,}\b', ' tokenphone ', text)
    
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 4. Interface Utilisateur & Style
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.pcworld.com/wp-content/uploads/2026/01/shutterstock_2495795811-2.jpg?quality=50&strip=all");
        background-size: cover;
        background-position: top center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='color: darkblue; text-align: center;'>📩 SMS Spam Detection System</h1>",
    unsafe_allow_html=True,
)

st.markdown("Enter a message below to check if it's spam or not.")

# 5. Champ de saisie et traitement
user_input = st.text_area("Enter SMS Text Here:")

SEUIL_SPAM = 0.35  

if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("Please enter a message to classify.")
    else:
        cleaned_text = advanced_preprocess(user_input)
        transformed_input = vectorizer.transform([cleaned_text])
        
        is_spam = False
        confidence = 0.0
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(transformed_input)[0]
            spam_prob = probs[1] if len(probs) > 1 else probs[0]
            confidence = spam_prob * 100
            if spam_prob >= SEUIL_SPAM:
                is_spam = True
        else:
            pred = model.predict(transformed_input)[0]
            is_spam = (pred in [1, "spam", "SPAM"])

        if is_spam:
            st.error(f"🚨 **SPAM DETECTED!** (Confiance : {confidence:.1f}%)\nPlease be cautious and avoid clicking on any links.")
        else:
            st.success(f"✅ **NOT SPAM** (Confiance Spam : {confidence:.1f}%)\nThis message is safe to read.")
