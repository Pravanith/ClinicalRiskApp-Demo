# train_model_pro.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import json
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, average_precision_score, log_loss
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

def generate_complex_data(n=5000):
    """
    Generates synthetic data with NON-LINEAR relationships and NOISE
    to simulate a real medical dataset (Harder for the model to learn).
    """
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 95, n),
        'inr': np.random.lognormal(mean=0, sigma=0.4, size=n), # Realistic skew
        'systolic_bp': np.random.normal(120, 15, n),
        'anticoagulant': np.random.randint(0, 2, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'liver_disease': np.random.randint(0, 2, n),
    })

    # Complex Interaction: High Risk = Old Age + Anticoagulant OR High INR + Liver
    # This is "Hidden" logic the model must discover
    risk_prob = (
        (df['age'] > 70) * (df['anticoagulant'] * 0.4) +
        (df['inr'] > 3.0) * 0.5 +
        (df['liver_disease'] * 0.3) + 
        np.random.normal(0, 0.1, n) # Random Biological Noise
    )
    
    # Target: 0 (Stable) or 1 (Critical Bleed Risk)
    df['target_bleed'] = (risk_prob > 0.4).astype(int)
    return df

def train():
    print("Training on synthetic demonstration data (not clinical validation)...")
    df = generate_complex_data()
    X = df.drop('target_bleed', axis=1)
    y = df['target_bleed']

    # 1. Define Features
    numeric_features = ['age', 'inr', 'systolic_bp']
    categorical_features = ['gender', 'anticoagulant', 'liver_disease']

    # 2. Create Preprocessing Pipeline (The "Secret Sauce")
    # This handles scaling and encoding automatically
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # 3. Create the Full Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])

    # 4. Hyperparameter Tuning (Stop Guessing!)
    param_dist = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.05, 0.1]
    }

    print("🧠 Tuning Model with 5-Fold Cross-Validation...")
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc',
        verbose=1,
        n_jobs=1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    search.fit(X_train, y_train)

    # Calibrate only with training folds. The holdout never participates in tuning.
    best_model = CalibratedClassifierCV(search.best_estimator_, method='sigmoid', cv=5)
    best_model.fit(X_train, y_train)
    probabilities = best_model.predict_proba(X_test)[:, 1]
    baseline = search.best_estimator_.predict_proba(X_test)[:, 1]
    def metrics(probs):
        return {"roc_auc": float(roc_auc_score(y_test, probs)),
                "average_precision": float(average_precision_score(y_test, probs)),
                "brier_score": float(brier_score_loss(y_test, probs)),
                "log_loss": float(log_loss(y_test, probs))}
    report = {"version": 4, "data_source": "synthetic", "clinical_validation": False,
              "seed": 42, "training_rows": len(X_train), "holdout_rows": len(X_test),
              "features": list(X.columns), "best_parameters": search.best_params_,
              "uncalibrated": metrics(baseline), "calibrated": metrics(probabilities),
              "classification": classification_report(y_test, best_model.predict(X_test), output_dict=True)}
    output = Path(__file__).resolve().parent
    joblib.dump(best_model, output / 'clinical_pipeline_v4.pkl')
    (output / 'model_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("Saved clinical_pipeline_v4.pkl and model_report.json; original model preserved.")

if __name__ == "__main__":
    train()
