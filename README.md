import re
import unicodedata
import joblib
import streamlit as st

# 1. Configuration de la page Streamlit
st.set_page_config(
    page_title="SMS Spam & Phishing Detector",
    page_icon="📩",
    layout="centered"
)

# 2. Fonctions de prétraitement du texte
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Normalisation Unicode (accents, etc.)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = text.lower()
    
    # Remplacement du leetspeak courant
    leetspeak = {'0': 'o', '1': 'i', '3': 'e', '4': 'a', '@': 'a', '$': 's', '!': 'i'}
    for char, replacement in leetspeak.items():
        text = text.replace(char, replacement)
        
    # Masquage des URLs et numéros de téléphone
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' URL_TOKEN ', text)
    text = re.sub(r'\b\d{7,}\b', ' PHONE_TOKEN ', text)
    
    # Suppression de la ponctuation inutile
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 3. Chargement des modèles sauvegardés (Adapté avec .pkl)
@st.cache_resource
def load_assets():
    try:
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        model = joblib.load("svm_model.pkl")
        return vectorizer, model
    except Exception as e:
        st.error(f"Erreur de chargement des modèles : {e}")
        return None, None

vectorizer, model = load_assets()

# 4. Interface Utilisateur Streamlit
st.title("📩 Real-Time SMS & Phishing Spam Detector")
st.markdown("Analyse de SMS en temps réel à l'aide du Traitement Automatique du Langage Naturel (NLP) et de Support Vector Machines (SVM).")

st.divider()

user_input = st.text_area(
    "Entrez le message SMS à analyser :",
    placeholder="Exemple: Claim your free $1000 gift card now at http://example.com or call 0800123456",
    height=120
)

if st.button("Predict / Analyser", type="primary"):
    if not user_input.strip():
        st.warning("Veuillez saisir un texte avant de lancer la prédiction.")
    elif vectorizer is None or model is None:
        st.error("Impossible d'effectuer la prédiction : les modèles ne sont pas chargés.")
    else:
        # Nettoyage et vectorisation
        cleaned_text = preprocess_text(user_input)
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Prédiction
        prediction = model.predict(text_vectorized)[0]
        
        # Probabilité (si supportée par le modèle)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(text_vectorized)[0]
            spam_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        else:
            spam_prob = None

        st.subheader("Résultat de l'analyse :")
        if prediction == 1 or prediction == "spam":
            st.error("🚨 **Alerte SPAM / PHISHING détecté !**")
            if spam_prob is not None:
                st.write(f"Probabilité de Spam : **{spam_prob * 100:.2f}%**")
        else:
            st.success("✅ **Message Légitime (HAM)**")
            if spam_prob is not None:
                st.write(f"Probabilité de Spam : **{spam_prob * 100:.2f}%**")
