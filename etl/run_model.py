# Trains XGBoost model to predict mortgage default risk.
# Usage: python etl/run_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import pickle
from dotenv import load_dotenv
import snowflake.connector
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

load_dotenv()
os.makedirs('models', exist_ok=True)
os.makedirs('dashboards', exist_ok=True)

def load_data_from_snowflake():
    print("Pulling data from Snowflake...")
    conn = snowflake.connector.connect(
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database='MORTGAGE_DB',
        schema='ANALYTICS'
    )
    query = """
    SELECT
        CREDIT_SCORE, ORIGINAL_DTI, ORIGINAL_LTV,
        ORIGINAL_INTEREST_RATE, ORIGINAL_LOAN_TERM,
        LOAN_AGE, LOAN_PURPOSE, PROPERTY_STATE, DEFAULT_FLAG
    FROM MORTGAGE_DB.ANALYTICS.FACT_LOAN_MONTHLY
    WHERE CREDIT_SCORE IS NOT NULL
      AND ORIGINAL_DTI IS NOT NULL
      AND ORIGINAL_LTV IS NOT NULL
    """
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"Loaded {len(df):,} rows")
    return df

def prepare_features(df):
    le_state   = LabelEncoder()
    le_purpose = LabelEncoder()
    df['PROPERTY_STATE_ENC'] = le_state.fit_transform(df['PROPERTY_STATE'].fillna('XX'))
    df['LOAN_PURPOSE_ENC']   = le_purpose.fit_transform(df['LOAN_PURPOSE'].fillna('U'))

    feature_cols = [
        'CREDIT_SCORE', 'ORIGINAL_DTI', 'ORIGINAL_LTV',
        'ORIGINAL_INTEREST_RATE', 'ORIGINAL_LOAN_TERM',
        'LOAN_AGE', 'LOAN_PURPOSE_ENC', 'PROPERTY_STATE_ENC'
    ]
    X = df[feature_cols]
    y = df['DEFAULT_FLAG']
    return X, y, feature_cols

def train_model(X, y):
    print("Training XGBoost model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos if pos > 0 else 1

    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale, random_state=42,
        eval_metric='logloss', early_stopping_rounds=20
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"\nAUC-ROC Score: {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=['Current', 'Default']))
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax,
                                      name=f'XGBoost (AUC={auc:.3f})')
    ax.set_title('ROC Curve — Mortgage Default Prediction')
    plt.tight_layout()
    plt.savefig('dashboards/roc_curve.png', dpi=150)
    plt.close()
    print("Saved: dashboards/roc_curve.png")
    return auc

def generate_shap_plots(model, X_test, feature_cols):
    print("\nGenerating SHAP plots...")
    X_sample  = X_test.sample(min(2000, len(X_test)), random_state=42)
    explainer  = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample,
                      feature_names=feature_cols,
                      plot_type="bar", show=False)
    plt.title("Top features driving mortgage default risk")
    plt.tight_layout()
    plt.savefig('dashboards/shap_feature_importance.png', dpi=150)
    plt.close()
    print("Saved: dashboards/shap_feature_importance.png")

    shap.summary_plot(shap_values, X_sample,
                      feature_names=feature_cols, show=False)
    plt.title("SHAP values — impact direction per feature")
    plt.tight_layout()
    plt.savefig('dashboards/shap_beeswarm.png', dpi=150)
    plt.close()
    print("Saved: dashboards/shap_beeswarm.png")

def save_model(model, feature_cols, auc):
    path = 'models/default_risk_model.pkl'
    with open(path, 'wb') as f:
        pickle.dump({'model': model, 'features': feature_cols, 'auc': auc}, f)
    print(f"\nModel saved to: {path}")

if __name__ == "__main__":
    df = load_data_from_snowflake()
    X, y, feature_cols = prepare_features(df)
    model, X_test, y_test = train_model(X, y)
    auc = evaluate_model(model, X_test, y_test)
    generate_shap_plots(model, X_test, feature_cols)
    save_model(model, feature_cols, auc)
    print(f"\nDone! Final AUC-ROC: {auc:.4f}")