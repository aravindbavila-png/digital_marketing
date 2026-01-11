# main_6features_xgb_tuned.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    f1_score
)
from xgboost import XGBClassifier
from joblib import dump

DATA_PATH = "digital_marketing_campaign_dataset.csv"

# -------------------- 1. LOAD & CLEAN DATA --------------------
df = pd.read_csv(DATA_PATH)

# Drop non-feature ID-like columns if they exist
df = df.drop(columns=["CustomerID", "AdvertisingPlatform", "AdvertisingTool"], errors="ignore")

# Ensure required columns exist
feature_cols = ["Age", "Gender", "Income", "AdSpend", "WebsiteVisits", "TimeOnSite"]
for col in feature_cols + ["Conversion"]:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in dataset")

# Keep only needed columns
df = df[feature_cols + ["Conversion"]].copy()

# Handle missing numeric values
for col in ["Age", "Income", "AdSpend", "WebsiteVisits", "TimeOnSite"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col].fillna(df[col].median(), inplace=True)

# -------------------- 2. ENCODE GENDER --------------------
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"].astype(str))

X = df[feature_cols].copy()
y = df["Conversion"].astype(int)

# -------------------- 3. TRAIN / TEST SPLIT --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------- 4. HANDLE CLASS IMBALANCE --------------------
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
if pos_count == 0:
    raise ValueError("No positive class in training data!")
scale_pos_weight = neg_count / pos_count

# -------------------- 5. BASE XGBOOST MODEL --------------------
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight
)

# -------------------- 6. GRID SEARCH (TUNING) --------------------
param_grid = {
    "n_estimators":     [100, 200],
    "max_depth":        [3, 4, 5],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    verbose=2
)


grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_


# -------------------- 7. EVALUATE WITH DEFAULT THRESHOLD 0.5 --------------------
y_proba_test = best_model.predict_proba(X_test)[:, 1]
y_pred_test_default = (y_proba_test >= 0.5).astype(int)

acc_default = accuracy_score(y_test, y_pred_test_default)
auc = roc_auc_score(y_test, y_proba_test)
f1_default = f1_score(y_test, y_pred_test_default)



# -------------------- 8. FIND BEST THRESHOLD (FOR F1 SCORE) --------------------
thresholds = np.linspace(0.1, 0.9, 81)  # 0.10, 0.11, ..., 0.90
best_f1 = -1
best_threshold = 0.5

for thr in thresholds:
    preds_thr = (y_proba_test >= thr).astype(int)
    f1 = f1_score(y_test, preds_thr)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thr



# -------------------- 9. SAVE MODEL, ENCODER & THRESHOLD --------------------
dump(best_model, "xgb_6features_best.joblib")
dump(le_gender, "gender_encoder.joblib")
dump(best_threshold, "xgb_6features_best_threshold.joblib")

