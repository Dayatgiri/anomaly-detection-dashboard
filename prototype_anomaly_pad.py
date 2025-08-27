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
def run_anomaly_detection(input_csv, contamination=0.05, random_state=42, date_format='%d/%m/%Y'):
    """Run Isolation Forest anomaly detection with minimal date parsing."""
    try:
        df = read_csv_cached(input_csv)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    st.write(f"Columns in dataset: {df.columns.tolist()}")

    # Clean the 'tanggal' column
    if 'tanggal' not in df.columns:
        st.error("The 'tanggal' column is missing in the dataset.")
        return None
    
    df['tanggal'] = df['tanggal'].str.strip()  # Strip any leading/trailing spaces

    # Try parsing dates with Indonesian format (dd/mm/yyyy)
    df['tanggal'] = pd.to_datetime(df['tanggal'], format=date_format, errors='coerce', dayfirst=True)

    # Check for invalid dates
    invalid_dates = df[df['tanggal'].isna()]
    if not invalid_dates.empty:
        st.warning(f"Found {len(invalid_dates)} invalid dates after parsing. They have been set as NaT.")
        
        # Log invalid rows for further review
        st.write("Invalid rows (with missing dates):")
        st.write(invalid_dates[['wp_id', 'tanggal']])

        # Option 1: Drop rows with invalid dates
        df = df.dropna(subset=['tanggal'])
        st.warning(f"Rows with invalid dates have been dropped. Remaining rows: {len(df)}")

    # Check if there's enough data to process
    if df.empty:
        st.error("The dataset is empty after filtering invalid dates. No data to process.")
        return None

    # Clean and transform columns
    df['rasio_pajakdibayar'] = df['rasio_pajakdibayar'].replace({',': '', '.': ''}, regex=True).astype(float)
    for col in ['omset', 'target_pajak', 'pajak_dibayar']:
        if col in df.columns:
            df[col] = df[col].replace({',': '', '.': ''}, regex=True).astype(float)

    # Calculate 'rasio_pajakdibayar'
    df['rasio_pajakdibayar'] = (df['pajak_dibayar'] / df['target_pajak']).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Extract month from 'tanggal'
    df['bulan'] = df['tanggal'].dt.month

    # Count transactions per wp_id and month
    counts = df.groupby(['wp_id', 'bulan']).size().rename('txn_wp_month').reset_index()
    df = df.merge(counts, on=['wp_id', 'bulan'], how='left')

    # One-hot encoding for 'kode_sector' and 'kode_kecamatan'
    df_encoded = pd.get_dummies(df, columns=['kode_sector', 'kode_kecamatan'], drop_first=True)

    # Define feature columns
    base_exclude = ['wp_id', 'tanggal', 'nama_sektor', 'nama_kecamatan']  # Drop these columns
    feature_cols = [c for c in df_encoded.columns if c not in base_exclude]

    X = df_encoded[feature_cols].copy()

    # Check if there are any valid rows to process
    if X.empty:
        st.error("There are no valid rows to scale. Please check your dataset.")
        return None

    # Log transform large numerical columns
    for col in ['pajak_dibayar', 'target_pajak', 'omset']:
        if col in X.columns:
            X[col] = np.log1p(X[col].clip(lower=0))

    # Standardize numerical features
    num_features = [c for c in ['pajak_dibayar', 'target_pajak', 'rasio_pajakdibayar', 'txn_wp_month', 'bulan', 'omset'] if c in X.columns]
    if num_features:
        X[num_features] = StandardScaler().fit_transform(X[num_features])

    # Train Isolation Forest model
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,  # Contamination defines the expected outliers proportion
        random_state=random_state,
        n_jobs=-1
    )
    pred = model.fit_predict(X)
    score = model.decision_function(X)

    # Prepare the final output dataframe
    out = df_encoded.copy()
    out['anomaly_label'] = pred
    out['anomaly_score'] = -score  # Larger score = more anomalous
    out['is_anomaly'] = out['anomaly_label'] == -1

    # Add sector and district names back
    out['nama_sektor'] = df['nama_sektor']
    out['nama_kecamatan'] = df['nama_kecamatan']

    return out

# =========================
# Visualization Function
# =========================
def create_visualizations(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Histogram of anomaly scores
    n, bins, patches = axes[0,0].hist(df['anomaly_score'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].set_xlabel('Anomaly Score')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title(f'Distribution of Anomaly Scores\n(Number of anomalies: {df["is_anomaly"].sum()})')

    # Annotate only non-zero bins with the count values and bin ranges
    for i in range(len(patches)):
        height = patches[i].get_height()
        if height > 0:  # Only annotate bins with data
            # Place text label at the top of the bar (count)
            axes[0,0].text(patches[i].get_x() + patches[i].get_width() / 2, height, str(int(height)),
                           ha='center', va='bottom', fontsize=10, color='black')
            
            # Add bin range label below the bar (only for bins with data)
            bin_range_label = f'{bins[i]:.2f} - {bins[i+1]:.2f}'
            axes[0,0].text(patches[i].get_x() + patches[i].get_width() / 2, -0.05 * max(n), bin_range_label,
                           ha='center', va='top', fontsize=9, color='black')

    # Remove x-tick labels for bin ranges to avoid duplication
    axes[0,0].set_xticks([])  # Clear x-ticks to avoid redundant range labels
    axes[0,0].tick_params(axis='x', pad=10)  # Adjust padding for better readability

    # Anomalies by sector
    anomalies = df[df['is_anomaly'] == True]
    sector_anomalies = anomalies['nama_sektor'].value_counts()
    axes[0,1].barh(sector_anomalies.index, sector_anomalies.values)
    axes[0,1].set_xlabel('Number of Anomalies')
    axes[0,1].set_title(f'Anomalies by Sector\n(Number of anomalies: {len(anomalies)})')

    # Paid vs Expected Pajak (scatter plot)
    sample_df = df.sample(min(1000, len(df)), random_state=42)
    if {'target_pajak','pajak_dibayar'}.issubset(sample_df.columns):
        x = (sample_df['target_pajak'].clip(lower=0) + 1)
        y = (sample_df['pajak_dibayar'].clip(lower=0) + 1)
        colors = ['red' if anom else 'blue' for anom in sample_df['is_anomaly']]
        axes[1,0].scatter(x, y, c=colors, alpha=0.6)
        axes[1,0].set_xlabel('Target Pajak (+1)')
        axes[1,0].set_ylabel('Pajak Dibayar (+1)')
        axes[1,0].set_title(f'Paid vs Expected Pajak (Red = Anomaly)\n(Number of anomalies: {sample_df["is_anomaly"].sum()})')
        axes[1,0].set_xscale('log')
        axes[1,0].set_yscale('log')

    # Anomalies over time
    if 'tanggal' in df.columns:
        if not np.issubdtype(df['tanggal'].dtype, np.datetime64):
            df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')

        if df['is_anomaly'].sum() > 0:
            time_anomalies = df.groupby(df['tanggal'].dt.to_period('M'))['is_anomaly'].mean()
            axes[1,1].plot(time_anomalies.index.astype(str), time_anomalies.values, marker='o')
            axes[1,1].set_xlabel('Time (Monthly)')
            axes[1,1].set_ylabel('Proportion of Anomalies')
            axes[1,1].set_title(f'Anomaly Proportion Over Time\n(Number of anomalies: {df["is_anomaly"].sum()})')
            plt.setp(axes[1,1].get_xticklabels(), rotation=45)
        else:
            axes[1,1].text(0.5, 0.5, 'No anomalies detected', ha='center')

    plt.tight_layout()
    return fig


# =========================
# Streamlit App
# =========================
def main():
    st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
    st.title("Anomaly Detection Dashboard")

    tab1, tab2, tab3 = st.tabs(["Environment Check", "Anomaly Detection", "Visualizations"])

    # Tab 1: Environment Check
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
        date_fmt = st.text_input("Date format (Python strftime)", value="%d/%m/%Y")
        if uploaded_file is not None and st.button("Run Detection"):
            with st.spinner("Running anomaly detection..."):
                df_out = run_anomaly_detection(uploaded_file, contamination=contamination, date_format=date_fmt)
                if df_out is not None:
                    anomalies = df_out[df_out["is_anomaly"]]
                    st.success(f"Detection complete! Found {len(anomalies)} anomalies.")
                    st.download_button(
                        "Download Full Results CSV",  # Allow download of both true and false anomalies
                        df_out.to_csv(index=False),  # Download the entire df_out
                        file_name="full_anomalies_results.csv",  # New file name for full results
                        mime="text/csv"
                    )
                    st.session_state.df = df_out

    # Tab 3: Visualizations
    with tab3:
        st.header("Result Visualizations")
        if 'df' in st.session_state:
            st.pyplot(create_visualizations(st.session_state.df))
            st.subheader("Top Anomalies")
            top_anomalies = st.session_state.df[st.session_state.df["is_anomaly"]].sort_values("anomaly_score", ascending=False).head(10)
            st.dataframe(top_anomalies)
            st.subheader("False Anomalies")
            false_anomalies = st.session_state.df[st.session_state.df["is_anomaly"] == False].sort_values("anomaly_score", ascending=False).head(10)
            st.dataframe(false_anomalies)
        else:
            st.info("Run anomaly detection first to see visualizations")

if __name__ == "__main__":
    main()
