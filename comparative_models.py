# comparative_models.py
# Compares KNN (baseline), 1D CNN and LSTM on the SMS Spam dataset.
# Fully self-contained: downloads the dataset automatically.

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------
# 1. LOAD DATA (automatic download)
# ---------------------------------------------------------------
URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(URL, sep="\t", header=None, names=["label", "message"])
X = df["message"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Labels as numbers for DL (1 = spam, 0 = ham)
y_train_num = (y_train == "spam").astype("float32")
y_test_num  = (y_test  == "spam").astype("float32")

# ---------------------------------------------------------------
# 2. SHARED TEXT VECTORIZATION (for CNN & LSTM)
# ---------------------------------------------------------------
MAX_TOKENS = 10000
SEQ_LEN    = 60
vectorize = layers.TextVectorization(max_tokens=MAX_TOKENS, output_sequence_length=SEQ_LEN)
vectorize.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(256))

# ---------------------------------------------------------------
# 3. KNN BASELINE
# ---------------------------------------------------------------
def build_knn():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("knn",   KNeighborsClassifier(n_neighbors=5, metric="cosine")),
    ])

# ---------------------------------------------------------------
# 4. 1D CNN
# ---------------------------------------------------------------
def build_cnn():
    model = keras.Sequential([
        keras.Input(shape=(1,), dtype="string"),
        vectorize,
        layers.Embedding(MAX_TOKENS, 64),
        layers.Conv1D(128, 5, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.5),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ---------------------------------------------------------------
# 5. LSTM
# ---------------------------------------------------------------
def build_lstm():
    model = keras.Sequential([
        keras.Input(shape=(1,), dtype="string"),
        vectorize,
        layers.Embedding(MAX_TOKENS, 64),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.5),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ---------------------------------------------------------------
# 6. EVALUATION
# ---------------------------------------------------------------
def evaluate(name, y_true, y_pred, y_prob, t=None):
    r = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "time_s": round(t, 2) if t is not None else None,
    }
    print(f"\n=== {name} ===")
    for k, v in r.items():
        if k != "model" and v is not None:
            print(f"{k:10s}: {v:.4f}" if isinstance(v, float) else f"{k:10s}: {v}")
    return r

results = []

# --- KNN ---
print("\n>> Training KNN...")
t0 = time.time(); knn = build_knn(); knn.fit(X_train, y_train)
knn_time = time.time() - t0
results.append(evaluate("KNN", y_test_num, knn.predict(X_test),
                        knn.predict_proba(X_test)[:, 1], knn_time))

# --- CNN ---
print("\n>> Training 1D CNN...")
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_num)).batch(32).shuffle(1000)
val_ds   = tf.data.Dataset.from_tensor_slices((X_test,  y_test_num)).batch(32)
t0 = time.time()
build_cnn().fit(train_ds, epochs=10, validation_data=val_ds,
                class_weight={0: 1.0, 1: 2.0}, verbose=0)
cnn = build_cnn()  # rebuild & retrain since we didn't store it
cnn.fit(train_ds, epochs=10, validation_data=val_ds,
        class_weight={0: 1.0, 1: 2.0}, verbose=0)
cnn_time = time.time() - t0
cnn_prob = cnn.predict(tf.data.Dataset.from_tensor_slices(X_test).batch(32), verbose=0).ravel()
results.append(evaluate("1D CNN", y_test_num, (cnn_prob >= 0.5).astype(int), cnn_prob, cnn_time))

# --- LSTM ---
print("\n>> Training LSTM...")
lstm = build_lstm()
t0 = time.time()
lstm.fit(train_ds, epochs=10, validation_data=val_ds,
         class_weight={0: 1.0, 1: 2.0}, verbose=0)
lstm_time = time.time() - t0
lstm_prob = lstm.predict(tf.data.Dataset.from_tensor_slices(X_test).batch(32), verbose=0).ravel()
results.append(evaluate("LSTM", y_test_num, (lstm_prob >= 0.5).astype(int), lstm_prob, lstm_time))

# ---------------------------------------------------------------
# 7. SUMMARY
# ---------------------------------------------------------------
print("\n\n===== COMPARATIVE SUMMARY =====")
print(pd.DataFrame(results).to_string(index=False))
