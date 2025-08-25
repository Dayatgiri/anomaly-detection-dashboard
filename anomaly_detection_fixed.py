import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

def install_required_packages():
    """Install required packages for the anomaly detection"""
    packages = [
        'pandas', 
        'numpy', 
        'scikit-learn', 
        'matplotlib', 
        'seaborn'
    ]
    
    print("Checking and installing required packages...")
    print("-" * 50)
    
    for package in packages:
        try:
            # Try to import the package to check if it's available
            if package == 'scikit-learn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"✗ {package} not found. Installing...")
            try:
                # Install the package using pip
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}. Please install it manually with: pip install {package}")
                return False
    return True

# Install packages before trying to import sklearn
if install_required_packages():
    try:
        from sklearn.ensemble import IsolationForest
        import seaborn as sns
        print("✓ All packages imported successfully!")
    except ImportError as e:
        print(f"✗ Import error after installation: {e}")
        print("Please try restarting your Python environment/kernel")
        sys.exit(1)
else:
    print("Failed to install all required packages. Please install them manually.")
    sys.exit(1)

def run_anomaly_detection(input_csv, contamination=0.06, random_state=42):
    """Run Isolation Forest anomaly detection"""
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv, parse_dates=["tanggal"])
    print(f"Loaded {len(df)} records")
    
    # Display column names to help debug
    print(f"Columns in dataset: {list(df.columns)}")
    
    # Recreate features
    sectors = sorted(df["sektor"].unique())
    kecamatan = sorted(df["kecamatan"].unique())
    sector_code = {s:i for i,s in enumerate(sectors)}
    kec_code = {k:i for i,k in enumerate(kecamatan)}
    
    # Create ratio column
    df["ratio_paid_expected"] = (df["paid_tax"] / df["expected_tax"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Extract month from date
    df["month"] = df["tanggal"].dt.month
    
    # Create transaction count per wp_id per month
    print("Calculating transaction counts per taxpayer per month...")
    counts = df.groupby(["wp_id", "month"]).size().reset_index(name="txn_wp_month")
    df = df.merge(counts, on=["wp_id", "month"], how="left")
    
    # Map sector and kecamatan to codes
    df["sector_code"] = df["sektor"].map(sector_code)
    df["kec_code"] = df["kecamatan"].map(kec_code)

    # Prepare features for model
    feature_columns = ["paid_tax", "expected_tax", "ratio_paid_expected", "month", "txn_wp_month", "sector_code", "kec_code"]
    
    # Check if all required columns exist
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns {missing_columns}. Using available columns only.")
        feature_columns = [col for col in feature_columns if col in df.columns]
    
    X = df[feature_columns].copy()
    
    # Add log-transformed features if the original columns exist
    if "paid_tax" in df.columns:
        X["log_paid"] = np.log1p(X["paid_tax"])
    if "expected_tax" in df.columns:
        X["log_expected"] = np.log1p(X["expected_tax"])
    
    # Drop original tax columns if log columns were created
    if "log_paid" in X.columns and "log_expected" in X.columns:
        X = X.drop(columns=["paid_tax", "expected_tax"], errors='ignore')

    print(f"Using {X.shape[1]} features for modeling: {list(X.columns)}")
    
    # Train Isolation Forest model
    print("Training Isolation Forest model...")
    model = IsolationForest(n_estimators=100, contamination=contamination, 
                           random_state=random_state, n_jobs=-1)
    pred = model.fit_predict(X)
    score = model.decision_function(X)
    df["anomaly_label"] = pred
    df["anomaly_score"] = -score  # Convert to positive where higher = more anomalous
    df["is_anomaly"] = df["anomaly_label"] == -1
    
    return df

def create_visualizations(df):
    """Create visualizations for the anomaly detection results"""
    print("Generating visualizations...")
    
    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Anomaly score distribution
    axes[0, 0].hist(df['anomaly_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Anomaly Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Anomaly Scores')
    if len(df) > 0:
        axes[0, 0].axvline(x=df['anomaly_score'].quantile(0.95), color='red', linestyle='--', label='95th percentile')
    axes[0, 0].legend()
    
    # Plot 2: Anomalies by sector (if sector column exists)
    if 'sektor' in df.columns:
        sector_anomalies = df.groupby('sektor')['is_anomaly'].mean().sort_values(ascending=False)
        axes[0, 1].barh(range(len(sector_anomalies)), sector_anomalies.values, color='salmon')
        axes[0, 1].set_yticks(range(len(sector_anomalies)))
        axes[0, 1].set_yticklabels(sector_anomalies.index)
        axes[0, 1].set_xlabel('Proportion of Anomalies')
        axes[0, 1].set_title('Anomaly Proportion by Sector')
    else:
        axes[0, 1].text(0.5, 0.5, 'Sector data not available', ha='center', va='center')
        axes[0, 1].set_title('Sector Data Not Available')
    
    # Plot 3: Paid vs Expected tax with anomalies highlighted (if columns exist)
    if 'paid_tax' in df.columns and 'expected_tax' in df.columns:
        sample_size = min(1000, len(df))
        sample_df = df.sample(sample_size, random_state=42) if len(df) > 1000 else df
        colors = ['red' if anom else 'blue' for anom in sample_df['is_anomaly']]
        axes[1, 0].scatter(sample_df['expected_tax'], sample_df['paid_tax'], c=colors, alpha=0.6)
        axes[1, 0].set_xlabel('Expected Tax')
        axes[1, 0].set_ylabel('Paid Tax')
        axes[1, 0].set_title('Paid vs Expected Tax (Red = Anomaly)')
        # Use log scale if there are large variations in tax values
        if sample_df['expected_tax'].max() / sample_df['expected_tax'].min() > 100:
            axes[1, 0].set_xscale('log')
            axes[1, 0].set_yscale('log')
    else:
        axes[1, 0].text(0.5, 0.5, 'Tax data not available', ha='center', va='center')
        axes[1, 0].set_title('Tax Data Not Available')
    
    # Plot 4: Anomalies over time (if date column exists)
    if 'tanggal' in df.columns:
        try:
            time_anomalies = df.groupby(df['tanggal'].dt.to_period('M'))['is_anomaly'].mean()
            axes[1, 1].plot(time_anomalies.index.astype(str), time_anomalies.values, marker='o')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Proportion of Anomalies')
            axes[1, 1].set_title('Anomaly Proportion Over Time')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        except:
            axes[1, 1].text(0.5, 0.5, 'Error processing time data', ha='center', va='center')
            axes[1, 1].set_title('Time Data Error')
    else:
        axes[1, 1].text(0.5, 0.5, 'Date data not available', ha='center', va='center')
        axes[1, 1].set_title('Date Data Not Available')
    
    plt.tight_layout()
    return fig

def main():
    """Main function to run the anomaly detection"""
    print("=" * 60)
    print("ANOMALY DETECTION SYSTEM")
    print("=" * 60)
    
    # Get input file from user or use default
    input_file = input("Enter path to CSV file (or press Enter for default): ").strip()
    if not input_file:
        input_file = "pad_dummy_transactions.csv"
        print(f"Using default file: {input_file}")
    
    try:
        # Run the anomaly detection
        df = run_anomaly_detection(input_file)
        
        # Count anomalies
        anomalies = df[df["is_anomaly"]]
        print(f"✓ Detection complete! Found {len(anomalies)} anomalies.")
        
        # Save results
        output_file = "anomaly_results.csv"
        anomalies.to_csv(output_file, index=False)
        print(f"✓ Results saved to: {output_file}")
        
        # Show basic statistics
        print(f"\nRESULTS SUMMARY")
        print("-" * 40)
        print(f"Total transactions: {len(df):,}")
        print(f"Anomalies detected: {len(anomalies):,}")
        print(f"Anomaly rate: {len(anomalies)/len(df)*100:.2f}%")
        
        if len(anomalies) > 0:
            # Show top anomalies by score
            print(f"\nTop 5 anomalies by score:")
            top_anomalies = anomalies.nlargest(5, 'anomaly_score')
            for i, row in top_anomalies.iterrows():
                anomaly_info = f"  WP {row['wp_id']}: Score={row['anomaly_score']:.4f}"
                if 'paid_tax' in row and 'expected_tax' in row:
                    anomaly_info += f", Paid={row['paid_tax']:,.0f}, Expected={row['expected_tax']:,.0f}"
                print(anomaly_info)
        
        # Create and show visualizations
        visualization = input("\nGenerate visualizations? (y/n): ").lower().strip()
        if visualization == 'y':
            fig = create_visualizations(df)
            plt.savefig('anomaly_visualizations.png', dpi=150, bbox_inches='tight')
            print("✓ Visualizations saved to 'anomaly_visualizations.png'")
            
            # Ask if user wants to show the plot
            show_plot = input("Show visualizations now? (y/n): ").lower().strip()
            if show_plot == 'y':
                plt.show()
                
    except FileNotFoundError:
        print(f"✗ Error: File '{input_file}' not found.")
        print("Please make sure the file exists and try again.")
    except Exception as e:
        print(f"✗ An error occurred: {str(e)}")
        print("\nFull error details:")
        traceback.print_exc()
        print("\nPlease check your data format and try again.")

if __name__ == "__main__":
    main()