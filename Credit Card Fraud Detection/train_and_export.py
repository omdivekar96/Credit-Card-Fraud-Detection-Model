"""
train_and_export.py
-------------------
Downloads the Kaggle Credit Card Fraud dataset via kagglehub,
trains a Random Forest with the same hyper-parameters from the report,
applies threshold=0.44, exports:
  - fraud_rf.joblib  (model + scaler + threshold + feature list)
  - sample_transactions.csv  (10 fraud + 10 legit raw rows for the demo)

Run once:  python train_and_export.py
"""

import os, pathlib, time, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ── 1. Load dataset ─────────────────────────────────────────────────
print("[*] Downloading dataset via kagglehub ...")
import kagglehub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_path = os.path.join(path, "creditcard.csv")
print(f"[OK] Dataset at: {csv_path}")

df = pd.read_csv(csv_path)
print(f"   Shape: {df.shape}  |  Fraud: {df.Class.sum()}  |  Legit: {(df.Class==0).sum()}")

# ── 2. Split ────────────────────────────────────────────────────────
X = df.drop(columns="Class")
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ── 3. Save raw X_test BEFORE scaling (for demo CSV) ───────────────
X_test_raw = X_test.copy()

# ── 4. Scale Time & Amount ──────────────────────────────────────────
scaler = StandardScaler()
X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
X_test[["Time", "Amount"]]  = scaler.transform(X_test[["Time", "Amount"]])

# ── 5. Train Random Forest (report hyper-params, depth-capped for size) ─
print("[*] Training Random Forest ...")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,              # cap depth → keeps .joblib well under 100 MB
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
print(f"[OK] Trained in {time.time()-t0:.1f}s")

# ── 6. Evaluate with threshold=0.44 ────────────────────────────────
THRESHOLD = 0.44
probs = rf.predict_proba(X_test)[:, 1]
y_pred = (probs >= THRESHOLD).astype(int)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
cm   = confusion_matrix(y_test, y_pred)

print(f"\n-- Metrics (threshold={THRESHOLD}) --")
print(f"   Accuracy:  {acc:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall:    {rec:.4f}")
print(f"   F1:        {f1:.4f}")
print(f"   Confusion matrix:\n{cm}\n")

# ── 7. Export sample_transactions.csv (raw, unscaled) ───────────────
combined = X_test_raw.assign(Class=y_test.values)
legit  = combined[combined.Class == 0].sample(10, random_state=42)
fraud  = combined[combined.Class == 1].sample(min(10, len(combined[combined.Class == 1])), random_state=42)
demo   = pd.concat([legit, fraud]).reset_index(drop=True)
out_dir = pathlib.Path(__file__).parent
demo.to_csv(out_dir / "sample_transactions.csv", index=False)
print(f"[OK] sample_transactions.csv  ({len(demo)} rows)")

# ── 8. Export model bundle ──────────────────────────────────────────
bundle = {
    "model": rf,
    "scaler": scaler,
    "threshold": THRESHOLD,
    "features": list(X_train.columns),
    "metrics": {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "confusion_matrix": cm.tolist(),
        "test_size": int(X_test.shape[0]),
    },
}
model_path = out_dir / "fraud_rf.joblib"
joblib.dump(bundle, model_path, compress=3)
size_mb = model_path.stat().st_size / (1024 * 1024)
print(f"[OK] fraud_rf.joblib  ({size_mb:.1f} MB)")

if size_mb > 100:
    print("[WARN] File > 100 MB - GitHub will reject it. Reduce n_estimators or max_depth.")
else:
    print("[OK] Under 100 MB - safe for GitHub (no Git LFS needed).")

print("\n[DONE] Now run:  streamlit run app.py")
