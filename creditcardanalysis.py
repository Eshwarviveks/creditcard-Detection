                                          #setup and data loading
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

# Load the dataset (download from Kaggle first)
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
df = pd.read_csv('creditcard.csv')

# Explore the data
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nClass distribution:")
print(df['Class'].value_counts(normalize=True))

                                 #Data Exploration and Preprocessing
# Visualize class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0: Genuine, 1: Fraud)')
plt.show()

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Scale the 'Amount' and 'Time' features
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Prepare features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

#Handling Class Imbalance with SMOTE
from imblearn.over_sampling import SMOTE # type: ignore

# Apply SMOTE only to the training set
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Check new class distribution
print("\nAfter SMOTE:")
print(pd.Series(y_res).value_counts())
#Model Training (XGBoost)
from xgboost import XGBClassifier # type: ignore
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix # type: ignore

# Calculate the scale_pos_weight parameter
fraud_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"\nFraud ratio: {fraud_ratio:.2f}")

# Train XGBoost model
xgb = XGBClassifier(scale_pos_weight=fraud_ratio, random_state=42)
xgb.fit(X_res, y_res)

# Predictions
y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:, 1]  # probabilities for ROC-AUC

# Evaluate
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Alternative Model: Autoencoder for Anomaly Detection
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# Prepare data (only genuine transactions for training)
X_train_normal = X_train[y_train == 0]

# Autoencoder architecture
input_dim = X_train_normal.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
encoder = Dropout(0.1)(encoder)
decoder = Dense(input_dim, activation='linear')(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(0.001), loss='mse')

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = autoencoder.fit(X_train_normal, X_train_normal,
                          epochs=50,
                          batch_size=256,
                          validation_split=0.1,
                          callbacks=[early_stop],
                          verbose=1)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training History')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Calculate reconstruction error
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
threshold = np.percentile(mse, 95)  # 95th percentile as threshold
y_pred_ae = [1 if e > threshold else 0 for e in mse]

# Evaluate autoencoder
print("\nAutoencoder Classification Report:")
print(classification_report(y_test, y_pred_ae))
print(f"Autoencoder ROC-AUC Score: {roc_auc_score(y_test, mse):.4f}")

#Real-time Simulation
import time

# Function to simulate real-time processing
def process_transaction(model, transaction):
    start_time = time.time()
    
    # Reshape for single prediction
    if len(transaction.shape) == 1:
        transaction = transaction.values.reshape(1, -1)
    
    # Predict
    prediction = model.predict(transaction)
    proba = model.predict_proba(transaction)[:, 1] if hasattr(model, 'predict_proba') else None
    
    processing_time = time.time() - start_time
    
    return prediction, proba, processing_time

# Test with XGBoost on first 100 test samples
processing_times = []
for i in range(100):
    _, _, pt = process_transaction(xgb, X_test.iloc[i])
    processing_times.append(pt)

print(f"\nAverage processing time: {np.mean(processing_times):.6f} seconds")
print(f"Max processing time: {np.max(processing_times):.6f} seconds")

# model comparison and selection
# Compare both models
print("XGBoost Performance:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

print("\nAutoencoder Performance:")
print(classification_report(y_test, y_pred_ae))
print(f"ROC-AUC: {roc_auc_score(y_test, mse):.4f}")

# Based on results, you might choose XGBoost for better performance
final_model = xgb

# Saving the Model for Deployment
import joblib

# Save the model and scaler
joblib.dump(final_model, 'credit_card_fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# To load later:
# model = joblib.load('credit_card_fraud_model.pkl')
# scaler = joblib.load('scaler.pkl')add