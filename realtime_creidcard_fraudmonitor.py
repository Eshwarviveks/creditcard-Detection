

import pandas as pd
import numpy as np
import time
from sklearn.ensemble import IsolationForest # type: ignore
from xgboost import XGBClassifier# type: ignore
from sklearn.preprocessing import StandardScaler# type: ignore
from collections import defaultdict

class FraudDetectionSystem:
    def __init__(self, model, threshold=0.3):  # Lowered default threshold
        self.model = model
        self.threshold = threshold
        self.stats = defaultdict(int)
        self.feature_columns = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']

    def prepare_features(self, tx):
        """Ensure consistent feature columns"""
        features = {col: tx[col] for col in self.feature_columns if col in tx}
        return pd.Series(features)

    def analyze_transaction(self, tx):
        """Process transaction with fraud detection"""
        self.stats['total'] += 1
        
        # Prepare features (exclude Class if present)
        tx_features = self.prepare_features(tx)
        
        # Treat negative amounts as highly suspicious
        if tx_features['Amount'] < 0:
            self.stats['negative_amount'] += 1
            return "HIGHLY_SUSPICIOUS", 0.95  # 95% confidence
            
        try:
            # Get fraud probability
            if hasattr(self.model, 'predict_proba'):
                prob = self.model.predict_proba([tx_features])[0][1]
            else:
                prob = 1/(1+np.exp(-self.model.decision_function([tx_features])))
            
            # Update stats
            if prob > self.threshold:
                self.stats['fraud'] += 1
                return "FRAUD", prob
            else:
                self.stats['clean'] += 1
                return "CLEAR", prob
                
        except Exception as e:
            self.stats['errors'] += 1
            return f"ERROR: {str(e)}", 0.0

    def display_dashboard(self, tx, status, prob):
        """Real-time monitoring interface"""
        print("\033c", end="")  # Clear console
        print(" LIVE FRAUD MONITORING")
        print("========================")
        print(f" Processed: {self.stats['total']} | "
              f" Clean: {self.stats['clean']} | "
              f" Fraud: {self.stats['fraud']}")
        print(f" Suspicious: {self.stats['negative_amount']} (Negative Amount) | "
              f" Errors: {self.stats['errors']}\n")
        
        print(f" Transaction ${tx['Amount']:.2f}")
        print("----------------------------")
        
        if status == "HIGHLY_SUSPICIOUS":
            print(f"\033[91mNEGATIVE AMOUNT (95% fraud confidence)\033[0m")
        elif status == "FRAUD":
            print(f" \033[91mFRAUD DETECTED ({prob:.1%} confidence)\033[0m")
        elif "ERROR" in status:
            print(f" \033[93m{status}\033[0m")
        else:
            print(f" \033[92mCLEAR ({prob:.1%} risk)\033[0m")

def load_data():
    """Load and preprocess data"""
    data = pd.read_csv('creditcard.csv')
    # Keep negative amounts for fraud detection
    data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    return data

def train_model(data):
    """Train model with balanced weights"""
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    model = XGBClassifier(
        scale_pos_weight=len(y[y==0])/len(y[y==1]),  # Auto-balance
        eval_metric='aucpr',
        n_estimators=150,
        max_depth=5,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

def simulate_transactions(data, model, sample_size=500):
    """Run enhanced simulation"""
    detector = FraudDetectionSystem(model, threshold=0.3)
    sample = data.sample(sample_size)
    
    print("Initializing fraud detection system...")
    time.sleep(2)
    
    try:
        for _, tx in sample.iterrows():
            status, prob = detector.analyze_transaction(tx)
            detector.display_dashboard(tx, status, prob)
            time.sleep(0.1 if status == "CLEAR" else 0.5)  # Pause longer for alerts
            
    except KeyboardInterrupt:
        print("\n Simulation stopped by user")
    
    finally:
        print("\n FINAL STATISTICS")
        print("=================")
        print(f"Total processed: {detector.stats['total']}")
        print(f"Clean transactions: {detector.stats['clean']}")
        print(f"Fraud detected: {detector.stats['fraud']}")
        print(f"Negative amounts: {detector.stats['negative_amount']}")
        print(f"Errors: {detector.stats['errors']}")

if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    
    print("\nTraining model...")
    model = train_model(data)
    
    print("\nStarting simulation (Ctrl+C to stop)...")
    time.sleep(2)
    simulate_transactions(data, model)