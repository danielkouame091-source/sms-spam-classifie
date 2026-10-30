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
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from data import FOLDER, load_split

keras.utils.set_random_seed(42)

X_train, X_test, y_train, y_test = load_split()
print(f"Training messages: {len(X_train)} | Test messages: {len(X_test)}")

# Labels as numbers: 1 = spam, 0 = ham
y_train_num = (y_train == "spam").astype("float32").to_numpy()

# Keep 10% of the training messages to watch the network while it learns
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train.to_numpy(),
    y_train_num,
    test_size=0.1,
    random_state=42,
    stratify=y_train_num,
)


def make_dataset(texts, labels=None, shuffle=False):
    """Pack the messages into a TensorFlow dataset (texts as 'string' tensors)."""
    texts = tf.constant([str(t) for t in texts], dtype=tf.string)
    texts = tf.reshape(texts, (-1, 1))
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(texts)
    else:
        labels = tf.constant(labels, dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        dataset = dataset.shuffle(len(texts), seed=42)
    return dataset.batch(32)


train_ds = make_dataset(X_fit, y_fit, shuffle=True)
val_ds = make_dataset(X_val, y_val)
test_ds = make_dataset(X_test)

# Layer that turns words into numbers (it learns the vocabulary from the training texts)
vectorize = layers.TextVectorization(max_tokens=10000, output_sequence_length=60)
vectorize.adapt(tf.data.Dataset.from_tensor_slices([str(t) for t in X_fit]).batch(256))

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
    train_ds,
    epochs=10,
    validation_data=val_ds,
    class_weight={0: 1.0, 1: 2.0},  # spam is rarer, so we give it more weight
    callbacks=[early_stop],
    verbose=2,
)
training_seconds = round(time.time() - start, 2)

# Evaluate on messages the model has never seen
probabilities = model.predict(test_ds, verbose=0).ravel()
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
