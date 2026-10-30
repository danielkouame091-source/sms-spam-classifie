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

# ==============================================================================
# 1. DATA INGESTION & HYGIENE (CHARGEMENT ET PRÉTRAITEMENT)
# ==============================================================================
print("Chargement et prétraitement des données...")
# Remplace par ton fichier de données (ex: IBM HR Analytics Employee Attrition)
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Encodage des variables catégorielles
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# Séparation des caractéristiques (X) et de la cible (y -> 'Attrition')
X = df.drop('Attrition', axis=1).values
y = df['Attrition'].values

# Division des données en ensembles d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Mise à l'échelle des caractéristiques (Essentiel pour KNN et les Réseaux de Neurones)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Échantillons d'entraînement : {len(X_train)}")
print(f"Échantillons de test : {len(X_test)}")

# ==============================================================================
# 2. MODÈLE BASELINE : MACHINE LEARNING (KNN)
# ==============================================================================
print("\n--- Entraînement du modèle baseline (KNN) ---")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"Précision KNN (Accuracy) : {acc_knn:.4f}")

# ==============================================================================
# 3. MODÈLE AVANCÉ : DEEP LEARNING (RÉSEAU DE NEURONES ARTIFICIELS - RNA)
# ==============================================================================
print("\n--- Entraînement du modèle Deep Learning (RNA) ---")
rna_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid') # Classification binaire (Attrition: Oui/Non)
])

rna_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle avec suivi de la validation
history = rna_model.fit(
    X_train_scaled, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    verbose=1
)

loss_rna, acc_rna = rna_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Précision RNA (Accuracy) : {acc_rna:.4f}")

# ==============================================================================
# 4. ÉVALUATION COMPARATIVE & RAPPORT DE PERFORMANCE
# ==============================================================================
print("\n" + "="*50)
print("RÉSUMÉ DU BENCHMARK DES PERFORMANCES - WORKFORCEIQ")
print("="*50)
print(f"Modèle Baseline (KNN) : Précision = {acc_knn:.4f}")
print(f"Modèle Profond (RNA)  : Précision = {acc_rna:.4f}")
print("="*50)

# Rapport de classification détaillé pour le RNA
y_pred_rna_prob = rna_model.predict(X_test_scaled)
y_pred_rna = (y_pred_rna_prob > 0.5).astype(int)

print("\n--- Rapport de Classification (Modèle RNA) ---")
print(classification_report(y_test, y_pred_rna, target_names=['No Attrition', 'Attrition']))

# Visualisation de la matrice de confusion
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred_rna)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non', 'Oui'], yticklabels=['Non', 'Oui'])
plt.title('Matrice de Confusion - WorkforceIQ (RNA)')
plt.xlabel('Prédiction')
plt.ylabel('Réalité')
plt.tight_layout()
plt.show()
