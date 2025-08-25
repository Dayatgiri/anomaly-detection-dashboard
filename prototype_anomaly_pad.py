import sys
import subprocess
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import streamlit as st

# ------------------------ Utilities ------------------------

def check_python_installation():
    """Check if Python is properly installed and accessible"""
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
    """Install required packages if missing"""
    packages = ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'streamlit']
    results = []
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            results.append(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError:
            results.append(f"✗ Failed to install {package}")
    return results

# ------------------------ Anomaly Detection ------------------------

def run_anomaly_detection(input_csv, contamination=0.06, random_state=42):
    """Run Isolation Forest anomaly detection with robust column handling"""
    df = pd.read_csv(input_csv, parse_dates=["tanggal"])

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # Check required columns
    required_cols = ["wp_id", "tanggal", "sektor", "kecamatan", "paid_tax", "expected_tax"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s) in CSV: {missing_cols}")

    # Derived features
    df["ratio_paid_expected"] = (df["paid_tax"] / df["expected_tax"]).replace([np.inf, -np.inf], 0)
    df["month"] = pd.to_datetime(df["tanggal"]).dt.month

    # Transactions per WP per month
    counts = df.groupby(["wp_id", "month"]).size().rename("txn_wp_month").reset_index()
    df = df.merge(counts, on=["wp_id", "month"], how="left")

    # Encode categorical columns
    df["sector_code"] = df["sektor"].map({s:i for i,s in enumerate(df["sektor"].unique())})
    df["kec_code"] = df["kecamatan"].map({k:i for i,k in enumerate(df["kecamatan"].unique())})

    # Select features for Isolation Forest
    feature_cols = ["paid_tax", "expected_tax", "ratio_paid_expected", "month",
                    "txn_wp_month", "sector_code", "kec_code"]
    X = df[feature_cols].copy()
    X["log_paid"] = np.log1p(X["paid_tax"])
    X["log_expected"] = np.log1p(X["expected_tax"])
    X = X.drop(columns=["paid_tax","expected_tax"])

    # Fit Isolation Forest
    model = IsolationForest(
        n_estimators=300, contamination=contamination, 
        random_state=random_state, n_jobs=-1
    )
    pred = model.fit_predict(X)
    score = model.decision_function(X)

    # Add anomaly results to DataFrame
    df["anomaly_label"] = pred
    df["anomaly_score"] = -score
    df["is_anomaly"] = df["anomaly_label"] == -1

    return df

# ------------------------ Visualizations ------------------------

def create_visualizations(df):
    """Create visualizations for the anomaly detection results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Anomaly score distribution
    axes[0, 0].hist(df['anomaly_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Anomaly Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Anomaly Scores')
    axes[0, 0].axvline(x=df['anomaly_score'].quantile(0.95), color='red', linestyle='--', label='95th percentile')
    axes[0, 0].legend()
    
    # Plot 2: Anomalies by sector
    sector_anomalies = df.groupby('sektor')['is_anomaly'].mean().sort_values(ascending=False)
    axes[0, 1].barh(range(len(sector_anomalies)), sector_anomalies.values, color='salmon')
    axes[0, 1].set_yticks(range(len(sector_anomalies)))
    axes[0, 1].set_yticklabels(sector_anomalies.index)
    axes[0, 1].set_xlabel('Proportion of Anomalies')
    axes[0, 1].set_title('Anomaly Proportion by Sector')
    
    # Plot 3: Paid vs Expected tax with anomalies highlighted
    sample_df = df.sample(min(1000, len(df)), random_state=42)
    colors = ['red' if anom else 'blue' for anom in sample_df['is_anomaly']]
    axes[1, 0].scatter(sample_df['expected_tax'], sample_df['paid_tax'], c=colors, alpha=0.6)
    axes[1, 0].set_xlabel('Expected Tax')
    axes[1, 0].set_ylabel('Paid Tax')
    axes[1, 0].set_title('Paid vs Expected Tax (Red = Anomaly)')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    
    # Plot 4: Anomalies over time
    time_anomalies = df.groupby(df['tanggal'].dt.to_period('M'))['is_anomaly'].mean()
    axes[1, 1].plot(time_anomalies.index.astype(str), time_anomalies.values, marker='o')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Proportion of Anomalies')
    axes[1, 1].set_title('Anomaly Proportion Over Time')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

# ------------------------ Streamlit App ------------------------

def main():
    st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
    st.title("Python Environment Setup & Anomaly Detection Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["Environment Check", "Anomaly Detection", "Visualizations"])
    
    # -------- Tab 1: Environment --------
    with tab1:
        st.header("Python Environment Check")
        st.write("""
        If you're getting "Python not installed" errors in VS Code, follow these steps:
        1. Make sure Python is installed from [python.org](https://python.org)
        2. Install the Python extension in VS Code
        3. Select the correct interpreter (Ctrl+Shift+P → "Python: Select Interpreter")
        """)
        
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

    # -------- Tab 2: Anomaly Detection --------
    with tab2:
        st.header("Run Anomaly Detection")
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        contamination = st.slider("Contamination parameter", 0.01, 0.5, 0.06, 0.01)
        
        if uploaded_file is not None and st.button("Run Detection"):
            with st.spinner("Running anomaly detection..."):
                try:
                    df = run_anomaly_detection(uploaded_file, contamination=contamination)
                    anomalies = df[df["is_anomaly"]]
                    st.success(f"Detection complete! Found {len(anomalies)} anomalies.")
                    
                    st.download_button(
                        label="Download Anomalies CSV",
                        data=anomalies.to_csv(index=False),
                        file_name="anomalies.csv",
                        mime="text/csv"
                    )
                    
                    st.session_state.df = df

                except Exception as e:
                    st.error(f"Error processing CSV: {e}")
                    st.info("Make sure your CSV contains columns: wp_id, tanggal, sektor, kecamatan, paid_tax, expected_tax")

    # -------- Tab 3: Visualizations --------
    with tab3:
        st.header("Result Visualizations")
        if 'df' in st.session_state:
            st.pyplot(create_visualizations(st.session_state.df))
            
            st.subheader("Top Anomalies")
            top_anomalies = st.session_state.df[st.session_state.df["is_anomaly"]].sort_values(
                "anomaly_score", ascending=False).head(10)
            st.dataframe(top_anomalies[['wp_id', 'sektor', 'kecamatan', 'paid_tax', 
                                        'expected_tax', 'anomaly_score']])
        else:
            st.info("Run anomaly detection first to see visualizations")

# ------------------------ Main Entry ------------------------

if __name__ == "__main__":
    main()
