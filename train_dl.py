# train_dl.py
# Trains the deep learning model: a neural network (Embedding + BiLSTM)
# built with TensorFlow / Keras. The text is turned into numbers by a
# TextVectorization layer that lives INSIDE the model, so the saved model
# accepts raw text directly.

import json
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow import keras
from tensorflow.keras import layers

from data import FOLDER, load_split

keras.utils.set_random_seed(42)

X_train, X_test, y_train, y_test = load_split()
print(f"Training messages: {len(X_train)} | Test messages: {len(X_test)}")

# The network needs text as an array of shape (n, 1) and labels as 0 / 1
x_train = np.array(X_train.tolist()).reshape(-1, 1)
x_test = np.array(X_test.tolist()).reshape(-1, 1)
y_train_num = (y_train == "spam").astype("float32").to_numpy()
y_test_num = (y_test == "spam").astype("float32").to_numpy()

# Layer that turns words into numbers (it learns the vocabulary from the training texts)
vectorize = layers.TextVectorization(max_tokens=10000, output_sequence_length=60)
vectorize.adapt(x_train.ravel())

# The neural network
model = keras.Sequential(
    [
        keras.Input(shape=(1,), dtype="string"),
        vectorize,
        layers.Embedding(input_dim=10000, output_dim=64),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.5),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Stop early if the network stops improving
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)

start = time.time()
model.fit(
    x_train,
    y_train_num,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    class_weight={0: 1.0, 1: 2.0},  # spam is rarer, so we give it more weight
    callbacks=[early_stop],
    verbose=2,
)
training_seconds = round(time.time() - start, 2)

# Evaluate on messages the model has never seen
probabilities = model.predict(x_test, verbose=0).ravel()
predictions = np.where(probabilities >= 0.5, "spam", "ham")
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, pos_label="spam")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"F1-score (spam): {f1:.4f}")
print(f"Training time: {training_seconds} s\n")
print(classification_report(y_test, predictions))

# Save the model and the results
model.save(FOLDER / "dl_model.keras")

with open(FOLDER / "results_dl.json", "w") as f:
    json.dump(
        {
            "model": "Neural network (deep learning)",
            "accuracy": round(float(accuracy), 4),
            "f1_spam": round(float(f1), 4),
            "training_seconds": training_seconds,
        },
        f,
        indent=2,
    )

print("Saved: dl_model.keras, results_dl.json")
