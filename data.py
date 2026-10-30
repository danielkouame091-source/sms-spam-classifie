# comparative_models.py
# Academic project: comparison of KNN (baseline), 1D-CNN and LSTM
# for SMS Spam classification.
#
# Place this file at the root of the repository, next to data.py and train_dl.py.
# Then replace `load_split_demo()` with `from data import load_split`.

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------
# 1. DATA LOADING (replace with your own load_split from data.py)
# ----------------------------------------------------------------------
def load_split_demo():
    """
    Dummy dataset used only to test the pipeline end-to-end.
    Replace this call with your real `load_split()` from data.py.
    """
    X = ["Free entry in 2 a wkly comp", "Ok lar... Joking wif u oni...",
         "WINNER!! You have won £1000", "I'm gonna be home soon",
         "Urgent! Call 09061701461", "Hey, how are you?"] * 50
    y = ["spam", "ham", "spam", "ham", "spam", "ham"] * 50
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- In your project, use this instead: ---
# from data import load_split
# X_train, X_test, y_train, y_test = load_split()

X_train, X_test, y_train, y_test = load_split_demo()

# ----------------------------------------------------------------------
# 2. COMMON PREPROCESSING
# ----------------------------------------------------------------------
# Binary encoding for deep learning models (1 = spam, 0 = ham)
y_train_num = (np.array(y_train) == "spam").astype("float32")
y_test_num  = (np.array(y_test)  == "spam").astype("float32")

# TextVectorization layer (same settings as in train_dl.py)
MAX_TOKENS = 10000
SEQ_LEN    = 60
vectorize = layers.TextVectorization(max_tokens=MAX_TOKENS,
                                     output_sequence_length=SEQ_LEN)
# Adapt only on the training set to avoid data leakage
vectorize.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(256))

# ----------------------------------------------------------------------
# 3. MODEL 1: KNN (BASELINE)
# ----------------------------------------------------------------------
def build_knn():
    """TF-IDF + KNN (k=5, cosine similarity)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("knn",   KNeighborsClassifier(n_neighbors=5, metric="cosine"))
    ])

# ----------------------------------------------------------------------
# 4. MODEL 2: 1D CNN
# ----------------------------------------------------------------------
def build_cnn():
    model = keras.Sequential([
        keras.Input(shape=(1,), dtype="string"),
        vectorize,
        layers.Embedding(input_dim=MAX_TOKENS, output_dim=64),
        layers.Conv1D(filters=128, kernel_size=5, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.5),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# ----------------------------------------------------------------------
# 5. MODEL 3: LSTM (BiLSTM, same as in train_dl.py)
# ----------------------------------------------------------------------
def build_lstm():
    model = keras.Sequential([
        keras.Input(shape=(1,), dtype="string"),
        vectorize,
        layers.Embedding(input_dim=MAX_TOKENS, output_dim=64),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.5),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# ----------------------------------------------------------------------
# 6. EVALUATION FUNCTION
# ----------------------------------------------------------------------
def evaluate_model(name, y_true, y_pred, y_prob=None, train_time=None):
    """Compute and print classification metrics."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)  # 1 = spam
    rec  = recall_score(y_true, y_pred, pos_label=1)
    f1   = f1_score(y_true, y_pred, pos_label=1)
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    if auc is not None:
        print(f"AUC      : {auc:.4f}")
    if train_time is not None:
        print(f"Training time: {train_time:.2f} s")
    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "auc": auc, "train_time": train_time}

# ----------------------------------------------------------------------
# 7. TRAINING AND COMPARATIVE EVALUATION
# ----------------------------------------------------------------------
results = []

# --- 7.1 KNN ---
print("Training KNN (baseline)...")
start = time.time()
knn = build_knn()
knn.fit(X_train, y_train)
knn_time = time.time() - start
y_pred_knn = knn.predict(X_test)
y_prob_knn = knn.predict_proba(X_test)[:, 1]
results.append(evaluate_model("KNN (TF-IDF)", y_test_num, y_pred_knn,
                              y_prob_knn, knn_time))

# --- 7.2 1D CNN ---
print("\nTraining 1D CNN...")
cnn = build_cnn()
train_ds_cnn = tf.data.Dataset.from_tensor_slices((X_train, y_train_num)).batch(32).shuffle(1000)
val_ds_cnn   = tf.data.Dataset.from_tensor_slices((X_test, y_test_num)).batch(32)
start = time.time()
history_cnn = cnn.fit(train_ds_cnn, epochs=10,
                      validation_data=val_ds_cnn,
                      class_weight={0: 1.0, 1: 2.0},
                      verbose=0)
cnn_time = time.time() - start
y_prob_cnn = cnn.predict(tf.data.Dataset.from_tensor_slices(X_test).batch(32), verbose=0).ravel()
y_pred_cnn = (y_prob_cnn >= 0.5).astype(int)
results.append(evaluate_model("1D CNN", y_test_num, y_pred_cnn,
                              y_prob_cnn, cnn_time))

# --- 7.3 LSTM (BiLSTM) ---
print("\nTraining LSTM (BiLSTM)...")
lstm = build_lstm()
train_ds_lstm = tf.data.Dataset.from_tensor_slices((X_train, y_train_num)).batch(32).shuffle(1000)
val_ds_lstm   = tf.data.Dataset.from_tensor_slices((X_test, y_test_num)).batch(32)
start = time.time()
history_lstm = lstm.fit(train_ds_lstm, epochs=10,
                        validation_data=val_ds_lstm,
                        class_weight={0: 1.0, 1: 2.0},
                        verbose=0)
lstm_time = time.time() - start
y_prob_lstm = lstm.predict(tf.data.Dataset.from_tensor_slices(X_test).batch(32), verbose=0).ravel()
y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)
results.append(evaluate_model("LSTM (BiLSTM)", y_test_num, y_pred_lstm,
                              y_prob_lstm, lstm_time))

# ----------------------------------------------------------------------
# 8. COMPARATIVE SUMMARY TABLE
# ----------------------------------------------------------------------
df_results = pd.DataFrame(results)
print("\n\n===== COMPARATIVE SUMMARY =====")
print(df_results.to_string(index=False))

# (Optional) save results for your report
# df_results.to_csv("model_comparison.csv", index=False)
