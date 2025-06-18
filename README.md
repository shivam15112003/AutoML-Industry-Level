# 🚀 AutoML Universal System

## 📌 Overview

This **Universal AutoML System** automates model detection, preprocessing, feature engineering, hyperparameter optimization, and model training for **classification, regression, and clustering** tasks. It includes full support for **deep learning (DNN)**, advanced preprocessing, and robust evaluation.

## ✨ Features

✅ **Task Detection** – Automatically detects if the task is supervised (classification/regression) or unsupervised (clustering).
✅ **Advanced Preprocessing** – Handles missing values, encodes categorical variables, removes outliers, scales features, and performs feature selection.
✅ **Class Imbalance Handling** – Applies **SMOTE** for imbalanced classification tasks.
✅ **Automated Model Selection** – Trains multiple models and selects the best one:

* Random Forest
* XGBoost
* LightGBM
* CatBoost
* DNN (Deep Neural Network)
  ✅ **Hyperparameter Optimization** – Uses **Optuna** for efficient hyperparameter tuning.
  ✅ **Unsupervised Learning** – Includes **KMeans, DBSCAN, GMM, Agglomerative Clustering** with Silhouette score evaluation.
  ✅ **Completely Modular** – Easy to modify, extend, and integrate additional models or tuning strategies.

## 🛠 Installation

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn imbalanced-learn lightgbm xgboost catboost tensorflow optuna joblib
```

## 📊 Usage

### 1️⃣ Prepare Dataset

* Provide your dataset as a CSV file.
* If supervised learning, ensure the dataset includes a target column.

### 2️⃣ Run AutoML

* Run the script:

```bash
python main.py
```

* You will be prompted to enter:

  * The dataset file path.
  * The target column name (if supervised task).

### 3️⃣ Output Files

* Models saved in `models/`
* Predictions saved in `reports/`
* Scaler saved in `models/`

## 🔎 Evaluation Metrics

* **Classification**: Accuracy
* **Regression**: Mean Squared Error
* **Unsupervised (Clustering)**: Silhouette Score

## 📂 Project Structure

```
📁 AutoML Universal System
├── main.py           # Main AutoML script
├── models/           # Saved models and scalers
├── reports/          # Prediction outputs
├── README.md         # Documentation
```

## ⚠ Notes

* Fully automated from dataset ingestion to model saving.
* Hyperparameter tuning is automatically invoked.
* Easily expandable to include additional models, metrics, or deployment capabilities.

---

This system delivers **complete end-to-end AutoML automation** for both **supervised and unsupervised machine learning tasks**, including deep learning integration.



# Universal AutoML System (Supervised & Unsupervised with DNN + Optuna)

This is a fully automated Machine Learning system that supports both **supervised** and **unsupervised** learning.  
It automatically performs:
- Data preprocessing (imputation, encoding, outlier removal, scaling)
- Feature selection
- Hyperparameter tuning using **Optuna**
- Traditional ML models (RandomForest, XGBoost, LightGBM, CatBoost)
- Deep Neural Network (DNN) using **TensorFlow**
- Unsupervised Clustering (KMeans, DBSCAN, GaussianMixture, Agglomerative)
- Model comparison & automatic best model selection
- Full reporting with model predictions and metrics

---

## Features

✅ Fully automatic supervised classification/regression  
✅ Fully automatic unsupervised clustering  
✅ Hyperparameter optimization using Optuna  
✅ Deep Learning (DNN) integration  
✅ Full preprocessing pipeline  
✅ Saves best models and reports automatically  
✅ Minimal user input required

---

## Installation

1️⃣ Clone this repository.

```bash
git clone https://github.com/your-repo/universal-automl.git
cd universal-automl
```bash
2️⃣ Install dependencies:
pip install -r requirements.txt
Or if you're using conda:
```bash
conda create -n automl python=3.11
conda activate automl
pip install -r requirements.txt



