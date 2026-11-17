"""
Credit Card Fraud Detection — Streamlit Prototype
──────────────────────────────────────────────────
Loads fraud_rf.joblib (Random Forest + scaler + threshold)
and sample_transactions.csv.  Two tabs:
  • Predict  – pick a sample, optionally edit Amount, get verdict
  • Model Results – static metrics & confusion matrix
"""

import streamlit as st
import pandas as pd
import joblib

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Prototype",
    page_icon="🛡️",
    layout="centered",
)

# ── Load model bundle & sample data (cached) ───────────────────────
@st.cache_resource
def load_bundle():
    return joblib.load("fraud_rf.joblib")

@st.cache_data
def load_samples():
    return pd.read_csv("sample_transactions.csv")

bundle  = load_bundle()
samples = load_samples()

model     = bundle["model"]
scaler    = bundle["scaler"]
threshold = bundle["threshold"]
features  = bundle["features"]
metrics   = bundle.get("metrics", {})

# ── Header ──────────────────────────────────────────────────────────
st.title("🛡️ Credit Card Fraud Detection")
st.caption(
    f"Random Forest  •  threshold **{threshold}**  •  "
    "trained on the Kaggle ULB dataset"
)

# ── Tabs ────────────────────────────────────────────────────────────
tab_predict, tab_results = st.tabs(["🔍 Predict", "📊 Model Results"])

# ━━━━━━━━━━━  Tab 1 — Predict  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_predict:
    st.subheader("Pick a sample transaction")
    st.write(
        "V1–V28 are PCA features that can't be typed manually, "
        "so the demo uses real rows from the test set."
    )

    idx = st.selectbox(
        "Transaction",
        samples.index,
        format_func=lambda i: (
            f"#{i}  —  Amount ${samples.loc[i, 'Amount']:.2f}  "
            f"({'🔴 Fraud' if samples.loc[i, 'Class'] else '🟢 Legit'})"
        ),
    )

    row = samples.drop(columns="Class").iloc[[idx]].copy()

    # Let the user tweak Amount
    new_amount = st.number_input(
        "Amount ($)", value=float(row["Amount"].iloc[0]), min_value=0.0, step=1.0
    )
    row["Amount"] = new_amount

    if st.button("Predict", type="primary", use_container_width=True):
        # Prepare features in training order, scale Time & Amount
        X = row[features].copy()
        X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])

        prob = model.predict_proba(X)[0, 1]

        # ── Verdict ─────────────────────────────────────────────────
        col_prob, col_verdict = st.columns([1, 1])
        with col_prob:
            st.metric("Fraud Probability", f"{prob:.2%}")
            st.progress(min(float(prob), 1.0))
        with col_verdict:
            st.metric("Threshold", f"{threshold:.2f}")
            if prob >= threshold:
                st.error("⚠️ **FRAUDULENT** transaction", icon="🚨")
            else:
                st.success("✅ **Legitimate** transaction", icon="✅")

        # Show actual label for verification
        actual = int(samples.loc[idx, "Class"])
        actual_label = "Fraud" if actual else "Legitimate"
        st.info(f"Actual label in test set: **{actual_label}**")

# ━━━━━━━━━━━  Tab 2 — Model Results  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_results:
    st.subheader("Performance on held-out test set")

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{metrics.get('accuracy',  99.95):.2f}%")
    c2.metric("Precision", f"{metrics.get('precision', 95.83):.2f}%")
    c3.metric("Recall",    f"{metrics.get('recall',    72.63):.2f}%")
    c4.metric("F1 Score",  f"{metrics.get('f1',        82.63):.2f}%")

    # Confusion matrix
    cm = metrics.get("confusion_matrix", [[56648, 3], [26, 69]])
    test_n = metrics.get("test_size", 56746)
    st.write(f"**Confusion matrix** (test set, {test_n:,} transactions)")
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual Legit", "Actual Fraud"],
            columns=["Pred Legit", "Pred Fraud"],
        ),
        use_container_width=True,
    )

    # Limitations
    missed = cm[1][0] if len(cm) > 1 else 26
    total_fraud = sum(cm[1]) if len(cm) > 1 else 95
    st.info(
        f"**Limitations:** {missed} of {total_fraud} frauds missed  •  "
        "offline dataset covering ~2 days  •  PCA-anonymised features"
    )

    st.divider()
    st.caption("Model: Random Forest (100 trees, max_depth=20)  •  scikit-learn 1.9.0")

