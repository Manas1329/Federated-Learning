import streamlit as st
import pandas as pd
import time
import os

st.set_page_config(page_title="Federated Learning Hub", layout="wide")
st.title("🏥 Clinical Federated Learning Aggregator Hub")
st.write("Monitoring global model optimization across 3 isolated hospital nodes.")

# Add run selection
run_mode = st.selectbox(
    "Select Run Mode to Monitor",
    ["No Quantization", "Quantized"]
)

suffix = "no_quantization" if run_mode == "No Quantization" else "quantized"
metrics_path = f"federated_healthcare/dashboard/results/metrics_{suffix}.csv"

# Fallback for root execution
if not os.path.exists(metrics_path):
    metrics_path = f"dashboard/results/metrics_{suffix}.csv"

# Placeholder for real-time metric visualization
chart_placeholder = st.empty()

while True:
    if os.path.exists(metrics_path):
        try:
            df = pd.read_csv(metrics_path)
            if not df.empty:
                with chart_placeholder.container():
                    st.subheader(f"Current Training Round: {df['Round'].iloc[-1]}")
                    
                    # Convert accuracy to percentage if it's <= 1.0
                    acc_series = df["Accuracy"]
                    if acc_series.max() <= 1.0:
                        acc_series = acc_series * 100
                        
                    chart_data = pd.DataFrame({
                        "Accuracy (%)": acc_series
                    }, index=df["Round"])
                    
                    st.line_chart(chart_data)
                    st.dataframe(df)
        except Exception as e:
            pass
    else:
        with chart_placeholder.container():
            st.info(f"Waiting for federated training rounds to start for {run_mode}...")
    time.sleep(2)