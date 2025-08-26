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
    return pd.read_csv(file)

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
def run_anomaly_detection(input_csv, contamination=0.05, random_state=42, date_format='%d/%m/%Y'):
    """Run Isolation Forest anomaly detection with robust parsing & feature handling"""
    try:
        df = read_csv_cached(input_csv)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

    # Standarisasi header
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    st.write(f"Columns in dataset: {df.columns.tolist()}")

    # Debug: check the first few rows of 'tanggal' before parsing
    st.write("Before parsing 'tanggal' column:")
    st.write(df[['tanggal']].head(10))

    # Cek kolom wajib minimum
    required_cols_min = ["wp_id", "tanggal", "kode_sector", "kode_kecamatan", "pajak_dibayar", "target_pajak"]
    missing_cols = [c for c in required_cols_min if c not in df.columns]
    if missing_cols:
        st.error(f"Missing required column(s): {missing_cols}")
        return None

    # Tanggal
    df['tanggal'] = pd.to_datetime(df['tanggal'], format=date_format, errors='coerce', dayfirst=True)

    # Debug: check the 'tanggal' column after parsing
    st.write("After parsing 'tanggal' column:")
    st.write(df[['tanggal']].head(10))

    # Handle invalid dates
    invalid_dates = df[df['tanggal'].isna()]
    if not invalid_dates.empty:
        st.warning(f"Found {len(invalid_dates)} invalid dates after parsing. They have been set as NaT.")

        # Log invalid rows for further review
        st.write("Invalid rows (with missing dates):")
        st.write(invalid_dates)

    # Perbaiki kolom rasio_pajakdibayar agar tidak ada titik ribuan dan menjadi format desimal
    df['rasio_pajakdibayar'] = df['rasio_pajakdibayar'].replace({',': '', '.': ''}, regex=True).astype(float)

    # Perbaiki kolom lainnya agar tidak ada titik ribuan
    for col in ['omset', 'target_pajak', 'pajak_dibayar']:
        if col in df.columns:
            df[col] = df[col].replace({',': '', '.': ''}, regex=True).astype(float)

    # Rasio dihitung ulang dan aman terhadap pembagi 0
    df['rasio_pajakdibayar'] = (df['pajak_dibayar'] / df['target_pajak']).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Bulan
    df['bulan'] = df['tanggal'].dt.month

    # txn_wp_month
    counts = df.groupby(['wp_id', 'bulan']).size().rename('txn_wp_month').reset_index()
    df = df.merge(counts, on=['wp_id', 'bulan'], how='left')

    # One-hot langsung untuk kode, TAPI tampilkan nama_sektor dan nama_kecamatan
    df_encoded = pd.get_dummies(df, columns=['kode_sector', 'kode_kecamatan'], drop_first=True)

    st.write(f"Columns after one-hot: {df_encoded.columns.tolist()}")

    # Fitur: sertakan variabel inti pajak & derived
    base_exclude = ['wp_id', 'tanggal', 'nama_sektor', 'nama_kecamatan']  # Drop sektor dan kecamatan nama dari fitur
    feature_cols = [c for c in df_encoded.columns if c not in base_exclude]

    X = df_encoded[feature_cols].copy()

    # Log transform variabel besaran uang (stabilkan skala)
    for col in ['pajak_dibayar', 'target_pajak', 'omset']:
        if col in X.columns:
            X[col] = np.log1p(X[col].clip(lower=0))

    # Standarisasi numerik terpilih (jika ada)
    num_features = [c for c in ['pajak_dibayar', 'target_pajak', 'rasio_pajakdibayar', 'txn_wp_month', 'bulan', 'omset'] if c in X.columns]
    if num_features:
        X[num_features] = StandardScaler().fit_transform(X[num_features])

    # Model
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    pred = model.fit_predict(X)
    score = model.decision_function(X)

    out = df_encoded.copy()
    out['anomaly_label'] = pred
    out['anomaly_score'] = -score  # lebih besar = lebih aneh
    out['is_anomaly'] = out['anomaly_label'] == -1

    # Kembalikan nama_sektor dan nama_kecamatan, ganti kode sektor dan kode kecamatan
    out['nama_sektor'] = df['nama_sektor']
    out['nama_kecamatan'] = df['nama_kecamatan']

    return out

# =========================
# Visualization Function
# =========================
def create_visualizations(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Distribusi skor
    axes[0,0].hist(df['anomaly_score'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].set_xlabel('Anomaly Score')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Distribution of Anomaly Scores')
    if df['anomaly_score'].notna().any():
        p95 = df['anomaly_score'].quantile(0.95)
        axes[0,0].axvline(x=p95, linestyle='--', label='95th percentile')
        axes[0,0].legend()

    # Anomali per sektor (berdasar nama_sektor yang dihidupkan)
    anomalies = df[df['is_anomaly'] == True]
    sector_anomalies = anomalies['nama_sektor'].value_counts()
    axes[0,1].barh(sector_anomalies.index, sector_anomalies.values)
    axes[0,1].set_xlabel('Number of Anomalies')
    axes[0,1].set_title('Anomalies by Sector')

    # Paid vs Target (log-safe)
    sample_df = df.sample(min(1000, len(df)), random_state=42)
    if {'target_pajak','pajak_dibayar'}.issubset(sample_df.columns):
        x = (sample_df['target_pajak'].clip(lower=0) + 1)
        y = (sample_df['pajak_dibayar'].clip(lower=0) + 1)
        colors = ['red' if anom else 'blue' for anom in sample_df['is_anomaly']]
        axes[1,0].scatter(x, y, c=colors, alpha=0.6)
        axes[1,0].set_xlabel('Target Pajak (+1)')
        axes[1,0].set_ylabel('Pajak Dibayar (+1)')
        axes[1,0].set_title('Paid vs Expected Pajak (Red = Anomaly)')
        axes[1,0].set_xscale('log')
        axes[1,0].set_yscale('log')
    else:
        axes[1,0].text(0.5, 0.5, 'target_pajak/pajak_dibayar not found', ha='center')

    # Anomali per waktu
    if 'tanggal' in df.columns:
        # Cek jika 'tanggal' sudah dalam format datetime
        if not np.issubdtype(df['tanggal'].dtype, np.datetime64):
            df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')

        # Pastikan ada data untuk anomali
        if df['is_anomaly'].sum() > 0:
            # Mengelompokkan data berdasarkan bulan
            time_anomalies = df.groupby(df['tanggal'].dt.to_period('M'))['is_anomaly'].mean()
            axes[1,1].plot(time_anomalies.index.astype(str), time_anomalies.values, marker='o')
            axes[1,1].set_xlabel('Time (Monthly)')
            axes[1,1].set_ylabel('Proportion of Anomalies')
            axes[1,1].set_title('Anomaly Proportion Over Time')
            plt.setp(axes[1,1].get_xticklabels(), rotation=45)
        else:
            axes[1,1].text(0.5, 0.5, 'No anomalies detected', ha='center')
    else:
        axes[1,1].text(0.5, 0.5, 'tanggal column missing', ha='center')

    plt.tight_layout()
    return fig

# =========================
# Streamlit App
# =========================
def main():
    st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
    st.title("Python Environment & Tax Anomaly Detection Dashboard")

    tab1, tab2, tab3 = st.tabs(["Environment Check", "Anomaly Detection", "Visualizations"])

    # Tab 1: Environment
    with tab1:
        st.header("Python Environment Check")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Check Python Installation"):
                status, message = check_python_installation()
                (st.success if status else st.error)(message)
        with col2:
            if st.button("Install Required Packages"):
                results = install_required_packages()
                for result in results:
                    (st.success if result.startswith("✓") else st.error)(result)

    # Tab 2: Anomaly Detection
    with tab2:
        st.header("Run Anomaly Detection")
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        contamination = st.slider("Contamination parameter", 0.01, 0.5, 0.05, 0.01)
        date_fmt = st.text_input("Date format (Python strftime)", value="%d/%m/%Y", help="Contoh: %d/%m/%Y")
        if uploaded_file is not None and st.button("Run Detection"):
            with st.spinner("Running anomaly detection..."):
                df_out = run_anomaly_detection(uploaded_file, contamination=contamination, date_format=date_fmt)
                if df_out is not None:
                    anomalies = df_out[df_out["is_anomaly"]]
                    st.success(f"Detection complete! Found {len(anomalies)} anomalies.")
                    st.download_button(
                        "Download Anomalies CSV",
                        anomalies.to_csv(index=False),
                        file_name="anomalies.csv",
                        mime="text/csv"
                    )
                    st.session_state.df = df_out

        # Opsi threshold manual (opsional)
        if 'df' in st.session_state:
            st.subheader("Optional: Manual anomaly threshold")
            thresh = st.slider(
                "Mark as anomaly if anomaly_score ≥ ...",
                float(st.session_state.df['anomaly_score'].min()),
                float(st.session_state.df['anomaly_score'].max()),
                float(st.session_state.df['anomaly_score'].quantile(0.95))
            )
            st.session_state.df['is_anomaly_manual'] = st.session_state.df['anomaly_score'] >= thresh
            st.write(
                f"Anomalies (manual threshold) = {int(st.session_state.df['is_anomaly_manual'].sum())}"
            )

    # Tab 3: Visualizations
    with tab3:
        st.header("Result Visualizations")
        if 'df' in st.session_state:
            st.pyplot(create_visualizations(st.session_state.df))
            st.subheader("Top Anomalies")
            cols_show = ['wp_id', 'pajak_dibayar', 'target_pajak', 'rasio_pajakdibayar', 'anomaly_score', 'nama_sektor', 'nama_kecamatan']
            top_anomalies = st.session_state.df[st.session_state.df["is_anomaly"]].sort_values(
                "anomaly_score", ascending=False).head(10)
            st.dataframe(top_anomalies[cols_show])
        else:
            st.info("Run anomaly detection first to see visualizations")

if __name__ == "__main__":
    main()
