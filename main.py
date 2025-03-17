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
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, silhouette_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.inspection import permutation_importance

# Reinforcement Learning Imports
import gym
from stable_baselines3 import PPO, DQN, A2C

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

# Detect Task Type
target_column = "Purchased"  # Change as per dataset
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

# Feature Selection using Permutation Importance
def select_features(model, X_train, y_train):
    model.fit(X_train, y_train)
    result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
    important_features = X.columns[np.argsort(result.importances_mean)[-10:]]  # Top 10 features
    return important_features

# Define Models
models = {
    "RandomForest": RandomForestClassifier() if task_type == "classification" else RandomForestRegressor(),
    "XGBoost": XGBClassifier() if task_type == "classification" else XGBRegressor(),
    "LightGBM": LGBMClassifier() if task_type == "classification" else LGBMRegressor(),
    "CatBoost": CatBoostClassifier(verbose=0) if task_type == "classification" else CatBoostRegressor(verbose=0),
    "GradientBoosting": GradientBoostingClassifier() if task_type == "classification" else GradientBoostingRegressor(),
    "LogisticRegression": LogisticRegression() if task_type == "classification" else None,
    "LinearRegression": None if task_type == "classification" else LinearRegression(),
    "PolynomialRegression": None if task_type == "classification" else PolynomialFeatures(degree=2),
    "DecisionTree": DecisionTreeClassifier() if task_type == "classification" else DecisionTreeRegressor(),
    "NaïveBayes_Gaussian": GaussianNB() if task_type == "classification" else None,
    "KMeans": KMeans(n_clusters=3) if task_type == "unsupervised" else None,
    "DBSCAN": DBSCAN() if task_type == "unsupervised" else None,
    "PCA": PCA(n_components=2) if task_type == "unsupervised" else None,
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

logging.info(f"Best Model Selected: {best_model._class.name_} with Score: {best_score:.2f}%")

# Hyperparameter Optimization
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5)
    }
    model = XGBClassifier(*params) if task_type == "classification" else XGBRegressor(*params)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    return accuracy_score(y_test, preds) if task_type == "classification" else -mean_squared_error(y_test, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
best_params = study.best_params

with open("models/best_params.json", "w") as f:
    json.dump(best_params, f)

best_model = XGBClassifier(*best_params) if task_type == "classification" else XGBRegressor(*best_params)
best_model.fit(X_train_scaled, y_train)
joblib.dump(best_model, "models/best_model.pkl")

# Train Reinforcement Learning Agent
env = gym.make("CartPole-v1")
agent = PPO("MlpPolicy", env, verbose=1)
agent.learn(total_timesteps=10000)
agent.save("models/reinforcement_agent")

logging.info("Full AutoML System Completed! 🚀")




