import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_missing_data(df, title, save_path):
    """
    Generates and saves a heatmap of missing values in the DataFrame.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Define paths (relative to the project root)
    raw_kyc_path = os.path.join("mnt", "data", "kyc.csv")
    cleaned_kyc_path = os.path.join("mnt", "output", "clean", "cleaned_kyc.csv")
    
    # Load the raw and cleaned KYC data
    try:
        raw_kyc = pd.read_csv(raw_kyc_path)
        cleaned_kyc = pd.read_csv(cleaned_kyc_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Print summary statistics for raw and cleaned KYC data
    print("Summary statistics for kyc.csv (raw):")
    print(raw_kyc.describe(include='all'))
    print("\nSummary statistics for kyc.csv (cleaned):")
    print(cleaned_kyc.describe(include='all'))

    # Create a folder to store the visuals (if it doesn't already exist)
    visuals_dir = "visuals_outputs"
    os.makedirs(visuals_dir, exist_ok=True)

    # Generate missing data heatmaps
    raw_missing_path = os.path.join(visuals_dir, "missing_raw_kyc.png")
    cleaned_missing_path = os.path.join(visuals_dir, "missing_clean_kyc.png")
    plot_missing_data(raw_kyc, "Missing Data Heatmap - Raw KYC", raw_missing_path)
    plot_missing_data(cleaned_kyc, "Missing Data Heatmap - Cleaned KYC", cleaned_missing_path)

    print(f"\nVisuals saved to folder: {visuals_dir}")

if __name__ == "__main__":
    main()
