import streamlit as st
import pickle
import io
import os
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")

MODEL_PATH = Path("xgb_model.pkl")
SCALER_PATH = Path("scaler.pkl")


def load_pickle_from_path_or_upload(path: Path, label: str):
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Failed loading {path.name}: {e}")
            return None
    uploaded = st.sidebar.file_uploader(f"Upload {label}", type=["pkl", "sav", "bin"]) 
    if uploaded is not None:
        try:
            return pickle.load(io.BytesIO(uploaded.read()))
        except Exception as e:
            st.sidebar.error(f"Failed to load uploaded file: {e}")
    return None


xgb_model = load_pickle_from_path_or_upload(MODEL_PATH, "XGBoost model (xgb_model.pkl)")
scaler = load_pickle_from_path_or_upload(SCALER_PATH, "Scaler (scaler.pkl)")

st.markdown("Enter transaction values below and click Predict.")

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Transaction Amount", min_value=0.0, value=0.0, format="%.2f")
    time = st.number_input("Transaction Time (seconds since first transaction)", min_value=0, value=0)
with col2:
    st.write("Derived features")
    # Allow user to override derived features if desired
    auto_derive = st.checkbox("Auto-derive hour/day from time", value=True)
    if auto_derive:
        hour = int((time // 3600) % 24)
        day = int(time // (3600*24))
        st.write("Hour:", hour)
        st.write("Day:", day)
    else:
        hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=0)
        day = st.number_input("Day (0+)", min_value=0, value=0)

amount_zscore = (amount - 88.0) / 250.0
high_value_flag = int(amount > 200)
amount_ratio = amount / 88.0 if 88.0 != 0 else 0.0

input_data = np.array([amount, time, hour, day, amount_zscore, high_value_flag, amount_ratio]).reshape(1, -1)

if st.button("Predict"):
    if xgb_model is None or scaler is None:
        st.error("Model or scaler not loaded. Upload them via the sidebar or place them in the app directory.")
    else:
        try:
            # Ensure input has the same number of features the scaler/model expect.
            expected_features = None
            if hasattr(scaler, "n_features_in_"):
                expected_features = int(getattr(scaler, "n_features_in_"))
            elif hasattr(xgb_model, "n_features_in_"):
                expected_features = int(getattr(xgb_model, "n_features_in_"))

            if expected_features is not None and expected_features != input_data.shape[1]:
                padded = np.zeros((1, expected_features), dtype=float)
                padded[0, : input_data.shape[1]] = input_data
                st.warning(f"Input has {input_data.shape[1]} features, but model expects {expected_features}. Padding remaining features with zeros for prediction.")
                use_input = padded
            else:
                use_input = input_data

            input_scaled = scaler.transform(use_input)
            prediction = xgb_model.predict(input_scaled)[0]
            st.success("⚠️ Fraudulent" if int(prediction) == 1 else "✅ Legitimate")
            st.write("Raw prediction:", int(prediction))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.sidebar.markdown("---")
st.sidebar.write("If you don't have the model files, place xgb_model.pkl and scaler.pkl in the app root, or upload them via the sidebar.")
# Prediction settings
st.sidebar.markdown("**Prediction settings**")
threshold = st.sidebar.slider("Fraud probability threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
show_importances = st.sidebar.checkbox("Show feature importances if available", value=False)

def get_prediction_and_proba(model, X):
    # Return (pred_label, prob) where prob is probability of class 1 if available else None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            prob1 = float(probs[:, 1][0])
            pred = 1 if prob1 >= threshold else 0
            return pred, prob1
        else:
            pred = int(model.predict(X)[0])
            return pred, None
    except Exception:
        # fallback to predict
        try:
            pred = int(model.predict(X)[0])
            return pred, None
        except Exception:
            return None, None

if show_importances and xgb_model is not None:
    if hasattr(xgb_model, "feature_importances_"):
        importances = list(xgb_model.feature_importances_)
        st.sidebar.write("Feature importances (first features):", importances[:10])
    else:
        st.sidebar.write("Model does not expose feature importances.")

# --- Finance settings: migrate to SQLite persistence, per-month thresholds, history and edit/delete
import sqlite3
from datetime import datetime

DB_PATH = Path("finance.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS thresholds (
            month_key TEXT PRIMARY KEY,
            threshold_pct REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL NOT NULL,
            note TEXT,
            timestamp TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

init_db()

def db_get_setting(key: str, default=None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cur.fetchone()
    conn.close()
    if row:
        return row[0]
    return default

def db_set_setting(key: str, value: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("REPLACE INTO settings(key, value) VALUES(?,?)", (key, str(value)))
    conn.commit()
    conn.close()

def db_set_threshold_for_month(month_key: str, pct: float):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("REPLACE INTO thresholds(month_key, threshold_pct) VALUES(?,?)", (month_key, float(pct)))
    conn.commit()
    conn.close()

def db_get_threshold_for_month(month_key: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT threshold_pct FROM thresholds WHERE month_key = ?", (month_key,))
    row = cur.fetchone()
    conn.close()
    return float(row[0]) if row else None

def db_add_transaction(amount_val: float, note: str = "", ts: str = None):
    if ts is None:
        ts = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO transactions(amount, note, timestamp) VALUES(?,?,?)", (float(amount_val), note, ts))
    conn.commit()
    last_id = cur.lastrowid
    conn.close()
    return last_id

def db_update_transaction(tx_id: int, amount_val: float, note: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE transactions SET amount = ?, note = ? WHERE id = ?", (float(amount_val), note, int(tx_id)))
    conn.commit()
    conn.close()

def db_delete_transaction(tx_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM transactions WHERE id = ?", (int(tx_id),))
    conn.commit()
    conn.close()

def db_get_transactions(month_key: str = None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    if month_key:
        # select transactions whose timestamp month matches
        cur.execute("SELECT id, amount, note, timestamp FROM transactions")
    else:
        cur.execute("SELECT id, amount, note, timestamp FROM transactions")
    rows = cur.fetchall()
    conn.close()
    if month_key:
        filtered = []
        for r in rows:
            try:
                dt = datetime.fromisoformat(r[3])
                k = f"{dt.year}-{dt.month}"
                if k == month_key:
                    filtered.append(r)
            except Exception:
                continue
        return filtered
    return rows

def db_get_month_spent(month_key: str):
    rows = db_get_transactions(month_key)
    total = 0.0
    for r in rows:
        total += float(r[1])
    return total

# UI: finance sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Finance settings**")
monthly_salary = float(db_get_setting("monthly_salary", 0.0))
fixed_monthly_expenses = float(db_get_setting("fixed_monthly_expenses", 0.0))
monthly_salary = st.sidebar.number_input("Monthly Salary", min_value=0.0, value=monthly_salary, format="%.2f")
fixed_monthly_expenses = st.sidebar.number_input("Fixed Monthly Expenses", min_value=0.0, value=fixed_monthly_expenses, format="%.2f")

current_year = datetime.utcnow().year
current_month = datetime.utcnow().month
current_month_key = f"{current_year}-{current_month}"

# Thresholds: global or per-month
global_warn_threshold = float(db_get_setting("warn_threshold_global", 10.0))
use_month_threshold = db_get_threshold_for_month(current_month_key) is not None
st.sidebar.markdown("**Warning threshold**")
threshold_mode = st.sidebar.selectbox("Threshold mode", options=["Global", "Per-month"], index=1 if use_month_threshold else 0)
if threshold_mode == "Global":
    warn_threshold_pct = st.sidebar.slider("Global warning threshold (% of salary remaining)", min_value=0, max_value=100, value=int(global_warn_threshold))
else:
    month_existing = db_get_threshold_for_month(current_month_key)
    warn_threshold_pct = st.sidebar.slider(f"Warning threshold for {current_month_key} (% of salary remaining)", min_value=0, max_value=100, value=int(month_existing) if month_existing is not None else 10)

if st.sidebar.button("Save finance settings"):
    db_set_setting("monthly_salary", monthly_salary)
    db_set_setting("fixed_monthly_expenses", fixed_monthly_expenses)
    if threshold_mode == "Global":
        db_set_setting("warn_threshold_global", warn_threshold_pct)
    else:
        db_set_threshold_for_month(current_month_key, warn_threshold_pct)
    st.sidebar.success("Finance settings saved.")

# Show current month spend / remaining
current_spent = db_get_month_spent(current_month_key)
remaining_budget = monthly_salary - fixed_monthly_expenses - current_spent
st.sidebar.markdown(f"**This month spent:** {current_spent:.2f}")
st.sidebar.markdown(f"**Remaining budget:** {remaining_budget:.2f}")

# Main UI: Add transaction, show history, edit/delete
st.markdown("---")
st.header("Transactions")

with st.expander("Add new transaction"):
    new_amount = st.number_input("Amount", min_value=0.0, value=0.0, format="%.2f", key="new_amount")
    new_note = st.text_input("Note", value="", key="new_note")
    if st.button("Add transaction"):
        tx_id = db_add_transaction(new_amount, new_note)
        st.success(f"Added transaction {tx_id}.")
        st.experimental_rerun()

# Transaction history view
all_months = set()
for r in db_get_transactions():
    try:
        dt = datetime.fromisoformat(r[3])
        all_months.add(f"{dt.year}-{dt.month}")
    except Exception:
        continue
months_sorted = sorted(list(all_months))
if current_month_key not in months_sorted:
    months_sorted.insert(0, current_month_key)

selected_month = st.selectbox("Filter by month", options=months_sorted, index=0 if months_sorted else 0)
rows = db_get_transactions(selected_month)
if rows:
    import pandas as pd

    df = pd.DataFrame(rows, columns=["id", "amount", "note", "timestamp"])
    st.dataframe(df)

    # Edit/delete selection
    selected_id = st.selectbox("Select transaction to edit/delete", options=df["id"].tolist())
    sel_row = df[df["id"] == selected_id].iloc[0]
    edit_amount = st.number_input("Edit amount", value=float(sel_row["amount"]), format="%.2f", key="edit_amount")
    edit_note = st.text_input("Edit note", value=str(sel_row["note"]), key="edit_note")
    col_edit, col_delete = st.columns(2)
    with col_edit:
        if st.button("Update transaction"):
            # require confirmation checkbox
            if st.checkbox("Confirm update", key="confirm_update"):
                db_update_transaction(int(selected_id), edit_amount, edit_note)
                st.success("Transaction updated.")
                st.experimental_rerun()
            else:
                st.info("Tick 'Confirm update' to apply changes.")
    with col_delete:
        if st.button("Delete transaction"):
            if st.checkbox("Confirm delete", key="confirm_delete"):
                db_delete_transaction(int(selected_id))
                st.success("Transaction deleted.")
                st.experimental_rerun()
            else:
                st.info("Tick 'Confirm delete' to remove transaction.")
else:
    st.info("No transactions for selected month.")

# Legacy buttons: finance warning & quick add (use DB functions)
if st.button("Show finance warning for this transaction"):
    projected_remaining = remaining_budget - float(amount)
    # determine effective threshold
    effective_threshold = None
    month_thresh = db_get_threshold_for_month(current_month_key)
    if month_thresh is not None:
        effective_threshold = float(month_thresh)
    else:
        effective_threshold = float(db_get_setting("warn_threshold_global", 10.0))

    if amount > remaining_budget:
        st.warning(f"This transaction of {amount:.2f} exceeds your remaining budget ({remaining_budget:.2f}).")
    elif projected_remaining <= 0:
        st.warning(f"This transaction would exhaust or exceed your monthly budget. Projected remaining: {projected_remaining:.2f}")
    else:
        pct_remaining = (projected_remaining / monthly_salary * 100) if monthly_salary > 0 else 100
        if pct_remaining <= effective_threshold:
            st.warning(f"After this transaction you'll have only {projected_remaining:.2f} left ({pct_remaining:.1f}% of salary).")
        else:
            st.success(f"Transaction looks within budget. Projected remaining: {projected_remaining:.2f}")

if st.button("Add this transaction to my monthly ledger"):
    db_add_transaction(amount, note="user transaction")
    st.success("Transaction added to ledger.")
    st.experimental_rerun()

# CSV export/import utilities
import io
import pandas as pd

def export_transactions_to_csv(month_key: str = None):
    rows = db_get_transactions(month_key)
    df = pd.DataFrame(rows, columns=["id", "amount", "note", "timestamp"] )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

st.markdown("---")
st.header("Import / Export")
col_e1, col_e2 = st.columns(2)
with col_e1:
    st.subheader("Export CSV")
    export_month = st.selectbox("Export month", options=months_sorted, index=0 if months_sorted else 0, key="export_month")
    if st.button("Download CSV"):
        csv_buf = export_transactions_to_csv(export_month)
        st.download_button(label="Download CSV", data=csv_buf.getvalue(), file_name=f"transactions_{export_month}.csv", mime="text/csv")
with col_e2:
    st.subheader("Import CSV")
    uploaded_csv = st.file_uploader("Upload transactions CSV (amount,note,timestamp optional)", type=["csv"]) 
    if uploaded_csv is not None:
        try:
            df_in = pd.read_csv(uploaded_csv)
            required = ["amount"]
            if not all(c in df_in.columns for c in required):
                st.error("CSV must contain at least an 'amount' column.")
            else:
                count = 0
                for _, r in df_in.iterrows():
                    amt = float(r.get("amount", 0.0))
                    note = str(r.get("note", "")) if "note" in r.index else "imported"
                    ts = None
                    if "timestamp" in r.index and pd.notna(r.get("timestamp")):
                        ts = str(r.get("timestamp"))
                    db_add_transaction(amt, note, ts)
                    count += 1
                st.success(f"Imported {count} transactions.")
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed to import CSV: {e}")
