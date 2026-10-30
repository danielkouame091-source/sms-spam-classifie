# train_ml.py
# Trains the machine learning model: TF-IDF + Linear SVM.

import json
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC

from data import FOLDER, load_split

X_train, X_test, y_train, y_test = load_split()
print(f"Training messages: {len(X_train)} | Test messages: {len(X_test)}")

# Turn the text into numbers
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the SVM
model = LinearSVC(random_state=42)
start = time.time()
model.fit(X_train_tfidf, y_train)
training_seconds = round(time.time() - start, 2)

# Evaluate on messages the model has never seen
predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, pos_label="spam")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"F1-score (spam): {f1:.4f}")
print(f"Training time: {training_seconds} s\n")
print(classification_report(y_test, predictions))

# Save the model, the vectorizer and the results
joblib.dump(model, FOLDER / "svm_model.pkl")
joblib.dump(vectorizer, FOLDER / "tfidf_vectorizer.pkl")

with open(FOLDER / "results_ml.json", "w") as f:
    json.dump(
        {
            "model": "SVM (machine learning)",
            "accuracy": round(accuracy, 4),
            "f1_spam": round(f1, 4),
            "training_seconds": training_seconds,
        },
        f,
        indent=2,
    )

print("Saved: svm_model.pkl, tfidf_vectorizer.pkl, results_ml.json")
