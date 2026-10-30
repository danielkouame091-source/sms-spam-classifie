import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ==========================================
# 1. DATA LOADING AND PREPROCESSING
# ==========================================
# Load dataset (IBM HR Analytics Employee Attrition dataset)
# Ensure you have the dataset file or use a public URL/local path
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Encode categorical variables
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# Separate features (X) and target (y -> 'Attrition')
X = df.drop('Attrition', axis=1).values
y = df['Attrition'].values

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling (crucial for KNN and Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ==========================================
# 2. MODEL 1: MACHINE LEARNING (KNN)
# ==========================================
print("\n--- Training KNN Model ---")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {acc_knn:.4f}")

# ==========================================
# 3. MODEL 2: DEEP LEARNING - ARTIFICIAL NEURAL NETWORK (RNA / MLP)
# ==========================================
print("\n--- Training Artificial Neural Network (RNA) ---")
rna_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid') # Binary classification (Attrition: Yes/No)
])

rna_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_rna = rna_model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1)

loss_rna, acc_rna = rna_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"RNA (Neural Network) Accuracy: {acc_rna:.4f}")

# ==========================================
# 4. COMPARATIVE EVALUATION & BENCHMARK
# ==========================================
print("\n==========================================")
print("PERFORMANCE BENCHMARK SUMMARY")
print("==========================================")
print(f"KNN (Scikit-Learn) : Accuracy = {acc_knn:.4f}")
print(f"RNA (Keras)        : Accuracy = {acc_rna:.4f}")
print("==========================================")

# Detailed classification report for RNA model
y_pred_rna_prob = rna_model.predict(X_test_scaled)
y_pred_rna = (y_pred_rna_prob > 0.5).astype(int)

print("\n--- RNA Classification Report ---")
print(classification_report(y_test, y_pred_rna, target_names=['No Attrition', 'Attrition']))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred_rna)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix - RNA Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
