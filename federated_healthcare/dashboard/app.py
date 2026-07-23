import streamlit as st
import pandas as pd
import time
import os

st.set_page_config(page_title="Federated Learning Hub", layout="wide")
st.title("🏥 Clinical Federated Learning Aggregator Hub")
st.write("Monitoring global model optimization across 3 isolated hospital nodes.")

metrics_path = "dashboard/metrics.csv"

# Placeholder for real-time metric visualization
chart_placeholder = st.empty()

while True:
    if os.path.exists(metrics_path):
        try:
            df = pd.read_csv(metrics_path)
            if not df.empty:
                with chart_placeholder.container():
                    st.subheader(f"Current Training Round: {df['Round'].iloc[-1]}")
                    st.line_chart(df.set_index("Round")["Accuracy"])
                    st.dataframe(df)
        except Exception as e:
            pass
    else:
        st.info("Waiting for federated training rounds to start...")
    time.sleep(2)