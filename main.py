import pandas as pd
import numpy as np
import joblib
import optuna
import shap
import os
import json
import logging
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.inspection import permutation_importance

# Setup Logging
logging.basicConfig(filename="logs/automl.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure necessary directories exist
for directory in ["models", "reports", "logs"]:
    os.makedirs(directory, exist_ok=True)

# Load Data
df = pd.read_csv("data/industry_data.csv")

# Data Preprocessing
def preprocess_data(df):
    logging.info("Preprocessing Data...")
    df.fillna(df.median(numeric_only=True), inplace=True)

    categorical_cols = df.select_dtypes(include=['object']).columns
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    joblib.dump(encoders, "models/encoders.pkl")
    return df

df = preprocess_data(df)

target_column = "Purchased"  # Update as per dataset
task_type = "classification" if df[target_column].nunique() <= 10 else "regression"
logging.info(f"Detected Task Type: {task_type.upper()}")

X = df.drop(columns=[target_column])
y = df[target_column]

# Handle Imbalanced Data for Classification
if task_type == "classification":
    X, y = SMOTE().fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")

# Convert scaled data back to DataFrame
X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# Define Models
models = {
    "RandomForest": RandomForestClassifier() if task_type == "classification" else RandomForestRegressor(),
    "XGBoost": XGBClassifier() if task_type == "classification" else XGBRegressor(),
    "LightGBM": LGBMClassifier() if task_type == "classification" else LGBMRegressor(),
    "CatBoost": CatBoostClassifier(verbose=0) if task_type == "classification" else CatBoostRegressor(verbose=0),
    "GradientBoosting": GradientBoostingClassifier() if task_type == "classification" else GradientBoostingRegressor(),
    "LogisticRegression": LogisticRegression() if task_type == "classification" else None,
    "DecisionTree": DecisionTreeClassifier() if task_type == "classification" else DecisionTreeClassifier(),
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500) if task_type == "classification" else None
}

# Model Evaluation with Cross-Validation
best_model, best_score = None, 0
cv = StratifiedKFold(n_splits=5) if task_type == "classification" else KFold(n_splits=5)

for name, model in models.items():
    if model:
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy" if task_type == "classification" else "neg_mean_squared_error")
        mean_score = np.mean(scores) * 100 if task_type == "classification" else np.mean(scores)
        
        logging.info(f"{name} CV Score: {mean_score:.2f}")
        if mean_score > best_score:
            best_score = mean_score
            best_model = model

logging.info(f"Best Model Selected: {best_model.__class__.__name__} with Score: {best_score:.2f}%")

# Hyperparameter Optimization
if best_score < 98:
    logging.info("Performing Hyperparameter Optimization...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5)
        }
        model = XGBClassifier(**params) if task_type == "classification" else XGBRegressor(**params)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        return accuracy_score(y_test, preds) if task_type == "classification" else -mean_squared_error(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_params

    with open("models/best_params.json", "w") as f:
        json.dump(best_params, f)

    best_model = XGBClassifier(**best_params) if task_type == "classification" else XGBRegressor(**best_params)
    best_model.fit(X_train_scaled, y_train)

    best_score = accuracy_score(y_test, best_model.predict(X_test_scaled)) if task_type == "classification" else 1 - mean_squared_error(y_test, best_model.predict(X_test_scaled))
    logging.info(f"Optimized Model Score: {best_score:.2f}%")

# Save the Best Model
joblib.dump(best_model, "models/best_model.pkl")

# Feature Importance Analysis
if hasattr(best_model, "feature_importances_"):
    feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": best_model.feature_importances_})
    feature_importance.sort_values(by="Importance", ascending=False).to_csv("reports/feature_importance.csv", index=False)

# SHAP Explainability
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test_df[:100])
shap.summary_plot(shap_values, X_test_df[:100])
plt.savefig("reports/shap_summary.png")

# Save Sample Predictions
sample = X_test_df[:10]
predictions = best_model.predict(sample)
results = pd.DataFrame(sample, columns=X.columns)
results["Prediction"] = predictions
results.to_excel("reports/predictions.xlsx", index=False)
print("\n✅ Predictions saved to 'predictions.xlsx'")

logging.info("AutoML Pipeline Completed Successfully! 🚀")




