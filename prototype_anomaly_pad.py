import sys
import subprocess
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import streamlit as st

# =========================
# Environment / Package Check
# =========================
def check_python_installation():
    try:
        result = subprocess.run([sys.executable, "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True, f"Python found: {result.stdout.strip()}"
        else:
            return False, "Python not found or not in PATH"
    except:
        return False, "Error checking Python installation"

def install_required_packages():
    packages = ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'streamlit']
    results = []
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            results.append(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError:
            results.append(f"✗ Failed to install {package}")
    return results

# =========================
# Anomaly Detection Function
# =========================
def run_anomaly_detection(input_csv, contamination=0.06, random_state=42):
    """Run Isolation Forest anomaly detection with safe column checks"""
    
    try:
        df = pd.read_csv(input_csv, parse_dates=["tanggal"])
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Required columns check
    required_cols = ["wp_id","tanggal","sektor","kecamatan","paid_tax","expected_tax"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required column(s) in CSV: {missing_cols}")
        return None

    # Create ratio and month
    if "ratio_paid_expected" not in df.columns:
        df["ratio_paid_expected"] = (df["paid_tax"] / df["expected_tax"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "month" not in df.columns:
        df["month"] = df["tanggal"].dt.month

    # txn_wp_month
    if "txn_wp_month" not in df.columns:
        counts = df.groupby(["wp_id","month"]).size().rename("txn_wp_month").reset_index()
        df = df.merge(counts, on=["wp_id","month"], how="left")

    # sector_code
    if "sector_code" not in df.columns:
        sectors = sorted(df["sektor"].unique())
        sector_code_map = {s:i for i,s in enumerate(sectors)}
        df["sector_code"] = df["sektor"].map(sector_code_map)

    # kec_code
    if "kec_code" not in df.columns:
        kecamatan = sorted(df["kecamatan"].unique())
        kec_code_map = {k:i for i,k in enumerate(kecamatan)}
        df["kec_code"] = df["kecamatan"].map(kec_code_map)

    # Feature selection
    try:
        X = df[["paid_tax", "expected_tax", "ratio_paid_expected", "month",
                "txn_wp_month", "sector_code", "kec_code"]].copy()
    except KeyError as e:
        st.error(f"Column missing for features: {e}")
        return df

    # Log transform
    X["log_paid"] = np.log1p(X["paid_tax"])
    X["log_expected"] = np.log1p(X["expected_tax"])
    X = X.drop(columns=["paid_tax","expected_tax"])

    # Isolation Forest
    model = IsolationForest(n_estimators=300, contamination=contamination, 
                            random_state=random_state, n_jobs=-1)
    pred = model.fit_predict(X)
    score = model.decision_function(X)
    
    df["anomaly_label"] = pred
    df["anomaly_score"] = -score
    df["is_anomaly"] = df["anomaly_label"] == -1

    return df

# =========================
# Visualization Function
# =========================
def create_visualizations(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Anomaly score distribution
    axes[0,0].hist(df['anomaly_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_xlabel('Anomaly Score')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Distribution of Anomaly Scores')
    axes[0,0].axvline(x=df['anomaly_score'].quantile(0.95), color='red', linestyle='--', label='95th percentile')
    axes[0,0].legend()

    # Anomalies by sector
    sector_anomalies = df.groupby('sektor')['is_anomaly'].mean().sort_values(ascending=False)
    axes[0,1].barh(range(len(sector_anomalies)), sector_anomalies.values, color='salmon')
    axes[0,1].set_yticks(range(len(sector_anomalies)))
    axes[0,1].set_yticklabels(sector_anomalies.index)
    axes[0,1].set_xlabel('Proportion of Anomalies')
    axes[0,1].set_title('Anomaly Proportion by Sector')

    # Paid vs Expected tax
    sample_df = df.sample(min(1000, len(df)), random_state=42)
    colors = ['red' if anom else 'blue' for anom in sample_df['is_anomaly']]
    axes[1,0].scatter(sample_df['expected_tax'], sample_df['paid_tax'], c=colors, alpha=0.6)
    axes[1,0].set_xlabel('Expected Tax')
    axes[1,0].set_ylabel('Paid Tax')
    axes[1,0].set_title('Paid vs Expected Tax (Red = Anomaly)')
    axes[1,0].set_xscale('log')
    axes[1,0].set_yscale('log')

    # Anomalies over time
    time_anomalies = df.groupby(df['tanggal'].dt.to_period('M'))['is_anomaly'].mean()
    axes[1,1].plot(time_anomalies.index.astype(str), time_anomalies.values, marker='o')
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('Proportion of Anomalies')
    axes[1,1].set_title('Anomaly Proportion Over Time')
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig

# =========================
# Main Streamlit App
# =========================
def main():
    st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
    st.title("Python Environment Setup & Anomaly Detection Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["Environment Check", "Anomaly Detection", "Visualizations"])
    
    # Tab 1: Environment
    with tab1:
        st.header("Python Environment Check")
        if st.button("Check Python Installation"):
            status, message = check_python_installation()
            if status:
                st.success(message)
            else:
                st.error(message)
        if st.button("Install Required Packages"):
            results = install_required_packages()
            for result in results:
                if result.startswith("✓"):
                    st.success(result)
                else:
                    st.error(result)
    
    # Tab 2: Anomaly Detection
    with tab2:
        st.header("Run Anomaly Detection")
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        contamination = st.slider("Contamination parameter", 0.01, 0.5, 0.06, 0.01)
        if uploaded_file is not None and st.button("Run Detection"):
            with st.spinner("Running anomaly detection..."):
                df = run_anomaly_detection(uploaded_file, contamination=contamination)
                if df is not None:
                    anomalies = df[df["is_anomaly"]]
                    st.success(f"Detection complete! Found {len(anomalies)} anomalies.")
                    st.download_button("Download Anomalies CSV", anomalies.to_csv(index=False),
                                       file_name="anomalies.csv", mime="text/csv")
                    st.session_state.df = df
    
    # Tab 3: Visualizations
    with tab3:
        st.header("Result Visualizations")
        if 'df' in st.session_state:
            st.pyplot(create_visualizations(st.session_state.df))
            st.subheader("Top Anomalies")
            top_anomalies = st.session_state.df[st.session_state.df["is_anomaly"]].sort_values(
                "anomaly_score", ascending=False).head(10)
            st.dataframe(top_anomalies[['wp_id','sektor','kecamatan','paid_tax','expected_tax','anomaly_score']])
        else:
            st.info("Run anomaly detection first to see visualizations")

if __name__ == "__main__":
    main()
