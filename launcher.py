import streamlit as st
import pandas as pd
import os
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from Script import train_model_from_in_progress, predict_next_from_prefix

IN_PROGRESS_CSV = "in_progress.csv"
TZ = ZoneInfo("Asia/Jakarta")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
    pd.DataFrame(columns=["Weekday", "Formation", "Time"]).to_csv(IN_PROGRESS_CSV, index=False)
    st.session_state["model"] = None

def append_row(formation_prefix):
    now = datetime.now(TZ)
    row = {
        "Weekday": now.strftime("%A"),
        "Formation": formation_prefix,
        "Time": now.strftime("%H:%M"),
    }
    pd.DataFrame([row]).to_csv(IN_PROGRESS_CSV, mode="a", header=False, index=False)

def undo_last_row():
    if not os.path.exists(IN_PROGRESS_CSV):
        return False, "No in_progress.csv yet."

    df = pd.read_csv(IN_PROGRESS_CSV)
    if df.empty:
        return False, "Nothing to undo."

    df = df.iloc[:-1]
    df.to_csv(IN_PROGRESS_CSV, index=False)
    return True, "Last entry removed."

st.title("HMM-based Perdict tool, for hoyoverse")

formation = st.selectbox("Choose Formation:", ["41", "42", "43", "44"])

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Confirm"):
        append_row(formation)
        st.session_state["model"] = train_model_from_in_progress(IN_PROGRESS_CSV)
        st.success(f"Saved: {formation}")

with col2:
    if st.button("Undo"):
        ok, msg = undo_last_row()
        if ok:
            st.session_state["model"] = train_model_from_in_progress(IN_PROGRESS_CSV)
            st.success(msg)
        else:
            st.warning(msg)

with col3:
    if st.button("Predict"):
        if not os.path.exists(IN_PROGRESS_CSV):
            st.warning("No in_progress.csv yet.")
        else:
            df = pd.read_csv(IN_PROGRESS_CSV)
            if len(df) < 2:
                st.warning("Need at least 2 inputs before prediction is meaningful.")
            else:
                model = st.session_state.get("model")
                if model is None:
                    model = train_model_from_in_progress(IN_PROGRESS_CSV)
                    st.session_state["model"] = model

                last_prefix = str(df["Formation"].iloc[-1])[:2]
                pred, probs = predict_next_from_prefix(model, last_prefix)

                st.success(f"Predicted next Formation: {pred}")
                prob_table = pd.DataFrame({
                    "value": ["41", "42", "43", "44"],
                    "probability": probs
                }).set_index("value")

                st.bar_chart(prob_table)

                st.dataframe(prob_table, use_container_width=True, hide_index=True)

st.subheader("In Progress Log")

if os.path.exists(IN_PROGRESS_CSV):
    df_log = pd.read_csv(IN_PROGRESS_CSV)
else:
    df_log = pd.DataFrame(columns=["Weekday", "Formation", "Time"])

st.dataframe(df_log, use_container_width=True)
