# 📩 Real-Time SMS & Phishing Spam Detector

An end-to-end Machine Learning web application designed to detect SMS spam and phishing attempts in real-time using Natural Language Processing (NLP) and Support Vector Machines (SVM).

## 🚀 Key Features

* **Advanced Text Preprocessing**: Handles leetspeak replacement, URL masking, and phone number tokenization.
* **SVM Classification Engine**: Optimized with TF-IDF vectorization and threshold tuning for high-precision fraud detection.
* **Interactive UI**: Clean Streamlit web interface with real-time probability display.

## 🛠️ Tech Stack

* **Language**: Python
* **ML & NLP**: Scikit-Learn, Joblib, Regex, Unicodedata
* **Frontend**: Streamlit

## 💻 How to Run Locally

```bash
# 1. Clone the repository
git clone [https://github.com/danielkouame091-source/sms-spam-classifie.git](https://github.com/danielkouame091-source/sms-spam-classifie.git)
cd sms-spam-classifie

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit
streamlit run app.py
