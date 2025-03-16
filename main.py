import pandas as pd
import numpy as np
import joblib
import optuna
import hyperopt
import openpyxl
import shap
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# Load Data
df = pd.read_csv("data/industry_data.csv")

# Data Preprocessing
def preprocess_data(df):
    print("Cleaning & Preprocessing Data...")
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

target_column = "Purchased"  # Change to match your dataset
task_type = "classification" if df[target_column].nunique() <= 10 else "regression"
print(f"Detected Task: {task_type.upper()}")

X = df.drop(columns=[target_column])
y = df[target_column]

if task_type == "classification":
    X, y = SMOTE().fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")

models = {
    "RandomForest": RandomForestClassifier() if task_type == "classification" else RandomForestRegressor(),
    "XGBoost": XGBClassifier() if task_type == "classification" else XGBRegressor(),
    "LightGBM": LGBMClassifier() if task_type == "classification" else LGBMRegressor(),
    "CatBoost": CatBoostClassifier(verbose=0) if task_type == "classification" else CatBoostRegressor(verbose=0),
    "GradientBoosting": GradientBoostingClassifier() if task_type == "classification" else GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression() if task_type == "classification" else None,
    "DecisionTree": DecisionTreeClassifier() if task_type == "classification" else DecisionTreeClassifier(),
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500) if task_type == "classification" else None
}

best_model, best_acc = None, 0
for name, model in models.items():
    if model:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions) if task_type == "classification" else 1 - mean_squared_error(y_test, predictions)
        print(f"{name} Performance: {acc*100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_model = model

print(f"Best Model: {best_model.__class__.__name__} with {best_acc*100:.2f}% Performance")

if best_acc < 0.98:
    print("Optimizing Model for Higher Accuracy...")
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5)
        }
        model = XGBClassifier(**params) if task_type == "classification" else XGBRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return accuracy_score(y_test, pred) if task_type == "classification" else -mean_squared_error(y_test, pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_params

    best_model = XGBClassifier(**best_params) if task_type == "classification" else XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)
    best_acc = accuracy_score(y_test, best_model.predict(X_test)) if task_type == "classification" else 1 - mean_squared_error(y_test, best_model.predict(X_test))
    print(f"Optimized Accuracy: {best_acc*100:.2f}%")

joblib.dump(best_model, "models/best_model.pkl")

explainer = shap.Explainer(best_model)
shap_values = explainer(X_test[:100])
shap.summary_plot(shap_values, X_test[:100])
plt.savefig("reports/shap_summary.png")

feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": best_model.feature_importances_})
feature_importance.sort_values(by="Importance", ascending=False).to_csv("reports/feature_importance.csv", index=False)

sample = X_test[:10]
predictions = best_model.predict(sample)
results = pd.DataFrame(sample, columns=X.columns)
results["Prediction"] = predictions
results.to_excel("predictions.xlsx", index=False)

print("Predictions saved to 'predictions.xlsx'")
