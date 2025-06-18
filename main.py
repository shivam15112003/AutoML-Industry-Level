import pandas as pd
import numpy as np
import os
import joblib
import logging
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from imblearn.over_sampling import SMOTE

import optuna
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# User Inputs
dataset = input('Enter dataset file path: ')
target = input('Enter target column name (leave blank if unsupervised): ').strip()

# Load dataset
df = pd.read_csv(dataset)
print("Dataset Loaded Successfully")

# Task type detection
if target and target in df.columns:
    supervised = True
    target_column = target
    print("Task: Supervised")
else:
    supervised = False
    print("Task: Unsupervised")

# Preprocessing
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Outlier removal (IQR)
Q1 = df_imputed.quantile(0.25)
Q3 = df_imputed.quantile(0.75)
IQR = Q3 - Q1
outlier_free = df_imputed[~((df_imputed < (Q1 - 1.5 * IQR)) | (df_imputed > (Q3 + 1.5 * IQR))).any(axis=1)]
print("Preprocessing Complete")

if supervised:
    X = outlier_free.drop(columns=[target_column])
    y = outlier_free[target_column]
    task_type = "classification" if y.nunique() <= 10 else "regression"
    print(f"Supervised Task Type: {task_type}")
else:
    X = outlier_free.copy()
    y = None
    task_type = "unsupervised"

# Scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")

# Feature Selection
if supervised:
    selector_model = RandomForestClassifier() if task_type == 'classification' else RandomForestRegressor()
    selector_model.fit(X_scaled, y)
    selector = SelectFromModel(selector_model, threshold="median")
    X_selected = selector.fit_transform(X_scaled, y)
else:
    X_selected = X_scaled

# Apply SMOTE if classification
if supervised and task_type == "classification":
    X_selected, y = SMOTE().fit_resample(X_selected, y)

# Hyperparameter tuning functions
def tune_model(trial, model_type):
    scoring = "accuracy" if task_type == "classification" else "neg_mean_squared_error"

    if model_type == "RandomForest":
        model = RandomForestClassifier() if task_type == "classification" else RandomForestRegressor()
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20)
        }
        model.set_params(**param)

    elif model_type == "XGBoost":
        model = XGBClassifier() if task_type == "classification" else XGBRegressor()
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3)
        }
        model.set_params(**param)

    elif model_type == "LightGBM":
        model = LGBMClassifier() if task_type == "classification" else LGBMRegressor()
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3)
        }
        model.set_params(**param)

    elif model_type == "CatBoost":
        model = CatBoostClassifier(verbose=0) if task_type == "classification" else CatBoostRegressor(verbose=0)
        param = {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3)
        }
        model.set_params(**param)

    score = cross_val_score(model, X_selected, y, cv=5, scoring=scoring).mean()
    return score

def tune_dnn(trial):
    scoring = "accuracy" if task_type == "classification" else "neg_mean_squared_error"
    neurons1 = trial.suggest_int("neurons1", 32, 256)
    neurons2 = trial.suggest_int("neurons2", 16, 128)
    lr = trial.suggest_float("learning_rate", 0.0001, 0.01)

    model = keras.Sequential([
        layers.Input(X_selected.shape[1]),
        layers.Dense(neurons1, activation='relu'),
        layers.Dense(neurons2, activation='relu'),
        layers.Dense(1, activation='sigmoid' if task_type == "classification" else 'linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy' if task_type == "classification" else 'mse')
    
    kfold = StratifiedKFold(n_splits=5) if task_type == "classification" else KFold(n_splits=5)
    scores = []
    for train_idx, val_idx in kfold.split(X_selected, y):
        model.fit(X_selected[train_idx], y[train_idx], epochs=10, batch_size=32, verbose=0)
        preds = model.predict(X_selected[val_idx])
        if task_type == "classification":
            preds = (preds > 0.5).astype(int)
            score = accuracy_score(y[val_idx], preds)
        else:
            score = -mean_squared_error(y[val_idx], preds)
        scores.append(score)

    return np.mean(scores)

if supervised:
    models = ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "DNN"]
    best_model_name, best_score = None, -np.inf if task_type == "classification" else np.inf

    for model_name in models:
        direction = "maximize" if task_type == "classification" else "minimize"
        study = optuna.create_study(direction=direction)
        if model_name == "DNN":
            study.optimize(tune_dnn, n_trials=10)
        else:
            study.optimize(lambda trial: tune_model(trial, model_name), n_trials=20)
        score = study.best_value
        if (task_type == "classification" and score > best_score) or (task_type == "regression" and score < best_score):
            best_score = score
            best_model_name = model_name
            best_params = study.best_params

    print("Best Model:", best_model_name)
    print("Best Score:", best_score)

    # Re-train best model
    if best_model_name == "DNN":
        neurons1 = best_params["neurons1"]
        neurons2 = best_params["neurons2"]
        lr = best_params["learning_rate"]
        final_model = keras.Sequential([
            layers.Input(X_selected.shape[1]),
            layers.Dense(neurons1, activation='relu'),
            layers.Dense(neurons2, activation='relu'),
            layers.Dense(1, activation='sigmoid' if task_type == "classification" else 'linear')
        ])
        final_model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                            loss='binary_crossentropy' if task_type == "classification" else 'mse')
        final_model.fit(X_selected, y, epochs=30, batch_size=32, verbose=1)
        final_model.save("models/final_model_dnn.keras")
        preds = (final_model.predict(X_selected) > 0.5).astype(int).flatten() if task_type == "classification" else final_model.predict(X_selected).flatten()
    else:
        if best_model_name == "RandomForest":
            final_model = RandomForestClassifier(**best_params) if task_type == "classification" else RandomForestRegressor(**best_params)
        elif best_model_name == "XGBoost":
            final_model = XGBClassifier(**best_params) if task_type == "classification" else XGBRegressor(**best_params)
        elif best_model_name == "LightGBM":
            final_model = LGBMClassifier(**best_params) if task_type == "classification" else LGBMRegressor(**best_params)
        elif best_model_name == "CatBoost":
            final_model = CatBoostClassifier(**best_params, verbose=0) if task_type == "classification" else CatBoostRegressor(**best_params, verbose=0)
        final_model.fit(X_selected, y)
        joblib.dump(final_model, f"models/final_model_{best_model_name}.pkl")
        preds = final_model.predict(X_selected)

    pd.DataFrame({"y_true": y, "y_pred": preds}).to_csv("reports/final_supervised_predictions.csv", index=False)

else:
    # Unsupervised AutoML
    cluster_algos = {
        "KMeans": KMeans(n_clusters=3),
        "DBSCAN": DBSCAN(eps=1.0, min_samples=5),
        "GMM": GaussianMixture(n_components=3),
        "Agglomerative": AgglomerativeClustering(n_clusters=3)
    }
    best_score = -np.inf
    best_labels = None
    best_algo = None
    
    for name, model in cluster_algos.items():
        labels = model.fit_predict(X_selected)
        score = silhouette_score(X_selected, labels)
        logging.info(f"{name} Silhouette Score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_labels = labels
            best_algo = name

    print("Best Clustering Algorithm:", best_algo)
    print("Best Silhouette Score:", best_score)
    pd.DataFrame({"Cluster_Label": best_labels}).to_csv("reports/final_unsupervised_predictions.csv", index=False)
