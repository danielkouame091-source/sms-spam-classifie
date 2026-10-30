import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense, Dropout

# ==========================================
# 1. DATA LOADING AND PREPROCESSING
# ==========================================
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])

# Convert labels: 'ham' -> 0, 'spam' -> 1
df['target'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['text'].values
y = df['target'].values

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ==========================================
# 2. MODEL 1: MACHINE LEARNING (KNN + TF-IDF)
# ==========================================
print("\n--- Training KNN Model ---")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_tfidf, y_train)

y_pred_knn = knn_model.predict(X_test_tfidf)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {acc_knn:.4f}")

# ==========================================
# 3. TEXT SEQUENCING FOR DEEP LEARNING
# ==========================================
MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 64

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

# ==========================================
# 4. MODEL 2: DEEP LEARNING - 1D CNN
# ==========================================
print("\n--- Training 1D CNN Model ---")
cnn_model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_cnn = cnn_model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_data=(X_test_pad, y_test), verbose=1)

loss_cnn, acc_cnn = cnn_model.evaluate(X_test_pad, y_test, verbose=0)
print(f"1D CNN Accuracy: {acc_cnn:.4f}")

# ==========================================
# 5. MODEL 3: DEEP LEARNING - LSTM / RNN
# ==========================================
print("\n--- Training LSTM / RNN Model ---")
lstm_model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    LSTM(units=64, return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_lstm = lstm_model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_data=(X_test_pad, y_test), verbose=1)

loss_lstm, acc_lstm = lstm_model.evaluate(X_test_pad, y_test, verbose=0)
print(f"LSTM Accuracy: {acc_lstm:.4f}")

# ==========================================
# 6. COMPARATIVE EVALUATION & BENCHMARK
# ==========================================
print("\n==========================================")
print("PERFORMANCE BENCHMARK SUMMARY")
print("==========================================")
print(f"KNN (Scikit-Learn) : Accuracy = {acc_knn:.4f}")
print(f"1D CNN (Keras)     : Accuracy = {acc_cnn:.4f}")
print(f"LSTM (Keras)       : Accuracy = {acc_lstm:.4f}")
print("==========================================")

# Detailed classification report for LSTM
y_pred_lstm_prob = lstm_model.predict(X_test_pad)
y_pred_lstm = (y_pred_lstm_prob > 0.5).astype(int)

print("\n--- LSTM Classification Report ---")
print(classification_report(y_test, y_pred_lstm, target_names=['Ham', 'Spam']))

#
