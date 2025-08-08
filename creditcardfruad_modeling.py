# Optimized Credit Card Fraud Detection
# Author: Data Scientist (Enhanced for Precision/Recall)
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split# type: ignore
from sklearn.preprocessing import StandardScaler# type: ignore
from sklearn.linear_model import LogisticRegression# type: ignore
from sklearn.ensemble import RandomForestClassifier, IsolationForest# type: ignore
from xgboost import XGBClassifier# type: ignore
from sklearn.metrics import (classification_report, confusion_matrix, # type: ignore
                            roc_auc_score, precision_recall_curve,
                            average_precision_score, RocCurveDisplay)# type: ignore
from imblearn.over_sampling import SMOTE # type: ignore
import time
import warnings
warnings.filterwarnings('ignore')

# ================== 1. Data Loading ==================
def load_data():
    try:
        data = pd.read_csv('creditcard.csv')
        print("✅ Data loaded successfully")
        print(f"📊 Shape: {data.shape} | Fraud Rate: {data['Class'].mean():.4f}")
        return data
    except FileNotFoundError:
        print("❌ Error: File 'creditcard.csv' not found")
        return None

# ================== 2. Data Preprocessing ================== 
def preprocess_data(data):
    # Scale only 'Amount' (other features are already PCA transformed)
    data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

# ================== 3. Handle Class Imbalance ==================
def balance_classes(X_train, y_train, method='smote'):
    print(f"\n🔄 Balancing classes using {method.upper()}...")
    
    if method == 'smote':
        smote = SMOTE(sampling_strategy=0.1, random_state=42)  # 10% fraud
        X_res, y_res = smote.fit_resample(X_train, y_train)
    else:  # Class weighting (no resampling)
        return X_train, y_train
    
    print(f"Class distribution after balancing:")
    print(pd.Series(y_res).value_counts())
    return X_res, y_res

# ================== 4. Model Training ==================
def train_models(X_train, y_train):
    models = {
        'XGBoost': XGBClassifier(
            scale_pos_weight=100,  # Fraud 100x more important
            eval_metric='aucpr',   # Optimize for Precision-Recall
            n_jobs=-1,
            random_state=42
        ),
        'Isolation Forest': IsolationForest(
            contamination=0.01,    # Expected fraud rate
            random_state=42
        )
    }
    
    print("\n🔧 Training models...")
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train if name != 'Isolation Forest' else X_train[y_train==0])
        print(f"{name} trained in {time.time()-start:.2f}s")
    
    return models

# ================== 5. Enhanced Evaluation ==================
def evaluate_model(model, X_test, y_test, threshold=0.5):
    if isinstance(model, IsolationForest):
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred == -1, 1, 0)  # Convert to 0/1
        y_prob = model.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-y_prob))  # Convert to probability-like
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > threshold).astype(int)
    
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
    
    print(f" ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f" AP Score: {average_precision_score(y_test, y_prob):.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Legit', 'Pred Fraud'],
                yticklabels=['True Legit', 'True Fraud'])
    plt.title('Confusion Matrix')
    plt.show()
    
    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label='Model')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()
    
    return y_prob

# ================== 6. Optimized Real-Time Simulation ==================
def simulate_detection(model, X_test, y_test, n=5, threshold=0.7):
    print(f"\n🔍 Simulating real-time detection (threshold={threshold})...")
    samples = X_test.sample(n, random_state=42)
    true_labels = y_test.loc[samples.index]
    
    results = []
    for i, (idx, sample) in enumerate(samples.iterrows()):
        if isinstance(model, IsolationForest):
            prob = 1 / (1 + np.exp(-model.decision_function(sample.values.reshape(1, -1))))
            pred = int(prob > threshold)
        else:
            prob = model.predict_proba(sample.values.reshape(1, -1))[0][1]
            pred = int(prob > threshold)
        
        results.append({
            'Sample': i+1,
            'True': 'Fraud' if true_labels[idx] else 'Legit',
            'Predicted': 'Fraud' if pred else 'Legit',
            'Confidence': f"{prob[0]:.2%}" if isinstance(prob, np.ndarray) else f"{prob:.2%}",
            'Correct': 'correct' if pred == true_labels[idx] else 'wrong'
        })
    
    print(pd.DataFrame(results).to_markdown(index=False))

# ================== MAIN EXECUTION ==================
if __name__ == "__main__":
    # 1. Load data
    data = load_data()
    if data is None:
        exit()
    
    # 2. Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # 3. Balance classes
    X_res, y_res = balance_classes(X_train, y_train, method='smote')
    
    # 4. Train models
    models = train_models(X_res, y_res)
    
    # 5. Evaluate
    print("\n" + "="*50)
    print("🧪 MODEL EVALUATION".center(50))
    print("="*50)
    
    for name, model in models.items():
        print(f"\n Evaluating {name}")
        y_prob = evaluate_model(model, X_test, y_test, threshold=0.7)
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        optimal_idx = np.argmax(precision + recall)
        optimal_threshold = thresholds[optimal_idx]
        print(f" Suggested threshold: {optimal_threshold:.4f}")
    
    # 6. Simulation
    print("\n" + "="*50)
    print("REAL-TIME SIMULATION".center(50))
    print("="*50)
    simulate_detection(models['XGBoost'], X_test, y_test, n=10)