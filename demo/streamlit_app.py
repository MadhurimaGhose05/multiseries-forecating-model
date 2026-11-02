import io, requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Multi-series forecasting", layout="wide")
st.title("Multi-series forecasting demo")
st.caption("Upload one hourly series. Forecasts are generated from its uploaded history.")
api = st.sidebar.text_input("API URL", "http://api:8000")
uploaded = st.file_uploader("Historical CSV", type=["csv"])
horizon = st.sidebar.slider("Forecast horizon (hours)", 1, 168, 24)
model = st.sidebar.selectbox("Model", ["global", "chronos"])
use_exog = st.sidebar.toggle("Use exogenous features", value=True)
interval = st.sidebar.selectbox("Interval display", [0.5, 0.9], format_func=lambda x: f"{int(x*100)}%")
if uploaded and st.button("Generate forecast", type="primary"):
    try:
        resp = requests.post(f"{api}/forecast",
            files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
            data={"horizon": horizon, "model": model, "interval_width": interval,
                  "use_exogenous": str(use_exog).lower()}, timeout=900)
        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text}")
        else:
            result = resp.json()
            hist = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            hist["timestamp"] = pd.to_datetime(hist.timestamp)
            fc = pd.DataFrame(result["forecasts"])
            fc["timestamp"] = pd.to_datetime(fc.timestamp)
            st.subheader(f"Forecast for {result['series_id']}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.timestamp, y=hist.target, name="History", line=dict(color="#64748b")))
            lo, hi = ("lower_50","upper_50") if interval == 0.5 else ("lower_90","upper_90")
            fig.add_trace(go.Scatter(x=fc.timestamp, y=fc[hi], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=fc.timestamp, y=fc[lo], fill="tonexty", fillcolor="rgba(59,130,246,0.18)",
                                     line=dict(width=0), name=f"{int(interval*100)}% interval"))
            fig.add_trace(go.Scatter(x=fc.timestamp, y=fc.prediction, name="Forecast", line=dict(color="#2563eb", width=2)))
            st.plotly_chart(fig, use_container_width=True)
            a,b,c = st.columns(3)
            a.metric("MAE", "N/A")
            b.metric("90% coverage", "N/A")
            c.metric("Model", model)
            st.info("MAE and empirical coverage require ground-truth future observations; they are not inferred from history.")
            st.download_button("Download forecast JSON", resp.content, "forecast.json", "application/json")
    except requests.RequestException as e:
        st.error(f"Could not reach forecasting API: {e}")
