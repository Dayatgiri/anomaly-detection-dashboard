import sys
import subprocess
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st

# =========================
# Utils
# =========================
@st.cache_data(show_spinner=False)
def read_csv_cached(file):
    return pd.read_csv(file)  # Use default comma delimiter

def parse_num_id(x):
    """
    Parse angka format Indonesia:
    - titik = ribuan → dihapus
    - koma = desimal → ganti titik
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    # lepas spasi dan karakter non-digit umum
    s = s.replace(" ", "")
    # jika ada titik dan koma, asumsikan id format
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        # contoh: "123.456" dan itu ribuan → hapus titik
        # jika ada >1 titik kecil kemungkinan desimal, tetap dihapus semuanya
        if s.count(".") > 1:
            s = s.replace(".", "")
    try:
        return float(s)
    except:
        return np.nan

# =========================
# Environment / Package Check
# =========================
def check_python_installation():
    try:
        result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True, f"Python found: {result.stdout.strip()}"
        else:
            return False, "Python not found or not in PATH"
    except Exception as e:
        return False, f"Error checking Python installation: {e}"

def install_required_packages():
    packages = ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'streamlit']
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
def run_anomaly_detection(input_csv, contamination=0.05, random_state=42):
    """Run Isolation Forest anomaly detection with minimal date parsing."""
    # Load dataset
    df = pd.read_csv(input_csv)

    # Preprocessing and Cleaning
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Check if 'tanggal' column exists
    if 'tanggal' not in df.columns:
        return "The 'tanggal' column is missing in the dataset."

    df['tanggal'] = pd.to_datetime(df['tanggal'], format='%d/%m/%Y', errors='coerce')

    # Clean the 'pajak_dibayar' and other numeric columns
    df['rasio_pajakdibayar'] = df['rasio_pajakdibayar'].replace({',': '', '.': ''}, regex=True).astype(float)
    
    # Clean and convert other relevant columns
    for col in ['omset', 'target_pajak', 'pajak_dibayar']:
        if col in df.columns:
            df[col] = df[col].replace({',': '', '.': ''}, regex=True).astype(float)

    # Calculate new 'rasio_pajakdibayar'
    df['rasio_pajakdibayar'] = (df['pajak_dibayar'] / df['target_pajak']).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Prepare features for anomaly detection
    df_encoded = pd.get_dummies(df, columns=['kode_sector', 'kode_kecamatan'], drop_first=True)

    feature_cols = [c for c in df_encoded.columns if c not in ['wp_id', 'tanggal', 'nama_sektor', 'nama_kecamatan']]
    X = df_encoded[feature_cols].copy()

    # Standardize numeric features
    num_features = [c for c in ['pajak_dibayar', 'target_pajak', 'rasio_pajakdibayar', 'omset'] if c in X.columns]
    if num_features:
        X[num_features] = StandardScaler().fit_transform(X[num_features])

    # Isolation Forest for anomaly detection
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=random_state, n_jobs=-1)
    pred = model.fit_predict(X)
    score = model.decision_function(X)

    # Add anomaly labels to the original dataframe
    df['anomaly_label'] = pred
    df['anomaly_score'] = -score
    df['is_anomaly'] = df['anomaly_label'] == -1

    return df

# =========================
# Visualizations
# =========================
def create_visualizations(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Distribution of Anomaly Scores
    axes[0, 0].hist(df['anomaly_score'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Anomaly Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Anomaly Scores')

    # Anomalies by Sector
    anomalies = df[df['is_anomaly'] == True]
    sector_anomalies = anomalies['nama_sektor'].value_counts()
    axes[0, 1].barh(sector_anomalies.index, sector_anomalies.values)
    axes[0, 1].set_xlabel('Number of Anomalies')
    axes[0, 1].set_title('Anomalies by Sector')

    # Paid vs Target Pajak (log scale)
    if {'target_pajak', 'pajak_dibayar'}.issubset(df.columns):
        x = (df['target_pajak'] + 1)
        y = (df['pajak_dibayar'] + 1)
        colors = ['red' if anom else 'blue' for anom in df['is_anomaly']]
        axes[1, 0].scatter(x, y, c=colors, alpha=0.6)
        axes[1, 0].set_xlabel('Target Pajak (+1)')
        axes[1, 0].set_ylabel('Pajak Dibayar (+1)')
        axes[1, 0].set_title('Paid vs Expected Pajak (Red = Anomaly)')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')

    # Anomalies over Time
    if 'tanggal' in df.columns:
        df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
        if df['is_anomaly'].sum() > 0:
            time_anomalies = df.groupby(df['tanggal'].dt.to_period('M'))['is_anomaly'].mean()
            axes[1, 1].plot(time_anomalies.index.astype(str), time_anomalies.values, marker='o')
            axes[1, 1].set_xlabel('Time (Monthly)')
            axes[1, 1].set_ylabel('Proportion of Anomalies')
            axes[1, 1].set_title('Anomaly Proportion Over Time')
            plt.setp(axes[1, 1].get_xticklabels(), rotation=45)

    plt.tight_layout()
    return fig

# =========================
# Streamlit App
# =========================
def main():
    st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
    st.title("Tax Anomaly Detection")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    contamination = st.slider("Contamination parameter", 0.01, 0.5, 0.05, 0.01)

    if uploaded_file is not None and st.button("Run Detection"):
        df_out = run_anomaly_detection(uploaded_file, contamination=contamination)
        if df_out is not None:
            anomalies = df_out[df_out['is_anomaly']]
            st.success(f"Detection complete! Found {len(anomalies)} anomalies.")
            st.download_button(
                "Download Anomalies CSV",
                anomalies.to_csv(index=False),
                file_name="anomalies.csv",
                mime="text/csv"
            )
            st.session_state.df = df_out

    # Visualization Tab
    if 'df' in st.session_state:
        st.pyplot(create_visualizations(st.session_state.df))

if __name__ == "__main__":
    main()
