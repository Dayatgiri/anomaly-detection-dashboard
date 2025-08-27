import sys
import subprocess
import pandas as pd
import numpy as np
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
# Anomaly Detection Function (Without Contamination and Isolation Forest)
# =========================
def run_anomaly_detection(input_csv, date_format='%d/%m/%Y'):
    """Run Anomaly detection based on target tax and paid tax using custom logic."""
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

    # Calculate rasio_pajakdibayar
    df['rasio_pajakdibayar'] = (df['pajak_dibayar'] / df['target_pajak']).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Logic for detecting anomalies based on pajak
    def deteksi_anomali(row):
        # If pajak dibayar > target pajak, it's an anomaly
        if row['pajak_dibayar'] > row['target_pajak']:
            return True
        # If pajak dibayar is less than target pajak by more than 10%
        elif row['pajak_dibayar'] < row['target_pajak'] and (row['target_pajak'] - row['pajak_dibayar']) / row['target_pajak'] > 0.1:
            return True
        # Otherwise, it's not an anomaly (within 10% tolerance)
        else:
            return False

    # Apply the anomaly detection function to each row
    df['is_anomaly'] = df.apply(deteksi_anomali, axis=1)

    # Add sector and district names back
    df['nama_sektor'] = df['nama_sektor']
    df['nama_kecamatan'] = df['nama_kecamatan']

    return df

# =========================
# Visualization Function
# =========================
def create_visualizations(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Histogram of anomalies (based on 'is_anomaly')
    n, bins, patches = axes[0, 0].hist(df['is_anomaly'].astype(int).dropna(), bins=2, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Anomaly Status (0=Normal, 1=Anomaly)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Distribution of Anomalies\n(Number of anomalies: {df["is_anomaly"].sum()})')

    # Annotate the histogram for better readability
    for i in range(len(patches)):
        height = patches[i].get_height()
        if height > 0:  # Only annotate bins with data
            axes[0, 0].text(patches[i].get_x() + patches[i].get_width() / 2, height, str(int(height)),
                            ha='center', va='bottom', fontsize=10, color='black')

    # Anomalies by sector
    anomalies = df[df['is_anomaly'] == True]
    sector_anomalies = anomalies['nama_sektor'].value_counts()
    bars = axes[0, 1].barh(sector_anomalies.index, sector_anomalies.values)
    axes[0, 1].set_xlabel('Number of Anomalies')
    axes[0, 1].set_title(f'Anomalies by Sector\n(Number of anomalies: {len(anomalies)})')

    # Annotate the anomalies by sector
    for bar in bars:
        width = bar.get_width()
        axes[0, 1].text(width, bar.get_y() + bar.get_height() / 2, str(int(width)), 
                        ha='left', va='center', fontsize=10, color='black')

    # Paid vs Expected Pajak (scatter plot) for all data, not just anomalies
    if {'target_pajak', 'pajak_dibayar'}.issubset(df.columns):
        x = (df['target_pajak'].clip(lower=0) + 1)  # Avoid division by zero or negative values
        y = (df['pajak_dibayar'].clip(lower=0) + 1)
        # Color code based on anomaly status
        colors = ['red' if anom else 'blue' for anom in df['is_anomaly']]
        scatter = axes[1, 0].scatter(x, y, c=colors, alpha=0.6)
        axes[1, 0].set_xlabel('Target Pajak (+1)')
        axes[1, 0].set_ylabel('Pajak Dibayar (+1)')
        axes[1, 0].set_title(f'Paid vs Expected Pajak (Red = Anomaly)\n(Number of anomalies: {df["is_anomaly"].sum()})')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')

        # Add legend with color description
        axes[1, 0].legend(['Normal (Blue)', 'Anomaly (Red)'], loc='upper right')

    # Anomalies over time (displaying count of anomalies)
    if 'tanggal' in df.columns:
        if not np.issubdtype(df['tanggal'].dtype, np.datetime64):
            df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')

        if df['is_anomaly'].sum() > 0:
            # Count anomalies per month
            time_anomalies = df.groupby(df['tanggal'].dt.to_period('M'))['is_anomaly'].sum()
            axes[1, 1].plot(time_anomalies.index.astype(str), time_anomalies.values, marker='o')
            axes[1, 1].set_xlabel('Time (Monthly)')
            axes[1, 1].set_ylabel('Number of Anomalies')
            axes[1, 1].set_title(f'Anomalies Count Over Time\n(Number of anomalies: {df["is_anomaly"].sum()})')
            plt.setp(axes[1, 1].get_xticklabels(), rotation=45)

            # Annotate the number of anomalies over time (showing the count)
            for i, count in enumerate(time_anomalies.values):
                x_pos = time_anomalies.index[i].strftime('%b-%Y')
                
                # Using annotate to show the count of anomalies
                axes[1, 1].annotate(f'{count}', 
                                    xy=(x_pos, count),  # Position the annotation at the point
                                    xytext=(0, 5),  # Small offset for text positioning
                                    textcoords='offset points',  # Text offset points to avoid overlap
                                    ha='center', va='bottom', fontsize=10, color='black', 
                                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

        else:
            axes[1, 1].text(0.5, 0.5, 'No anomalies detected', ha='center')

    # Add legends for scatter plot and other relevant parts
    axes[0, 1].legend(['Anomalies per Sector'], loc='upper right')
    axes[1, 0].legend(['Anomalies (Red) vs Normal (Blue)', 'Anomaly Threshold'], loc='upper left')

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
        date_fmt = st.text_input("Date format (Python strftime)", value="%d/%m/%Y")
        if uploaded_file is not None and st.button("Run Detection"):
            with st.spinner("Running anomaly detection..."):
                df_out = run_anomaly_detection(uploaded_file, date_format=date_fmt)
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
            top_anomalies = st.session_state.df[st.session_state.df["is_anomaly"]].sort_values("is_anomaly", ascending=False).head(10)
            st.dataframe(top_anomalies)
            st.subheader("False Anomalies")
            false_anomalies = st.session_state.df[st.session_state.df["is_anomaly"] == False].sort_values("is_anomaly", ascending=False).head(10)
            st.dataframe(false_anomalies)
        else:
            st.info("Run anomaly detection first to see visualizations")

if __name__ == "__main__":
    main()
