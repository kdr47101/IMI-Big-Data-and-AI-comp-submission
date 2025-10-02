import os
import logging
import pandas as pd
from sklearn.ensemble import IsolationForest

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define output directory for anomaly detection results
TASK1_OUTPUT_DIR = os.path.join("mnt", "output", "task1")
os.makedirs(TASK1_OUTPUT_DIR, exist_ok=True)

def detect_anomalies(df, contamination=0.01, features=None):
    """
    Detect anomalies in the DataFrame using IsolationForest.
    
    Parameters:
      df (pd.DataFrame): DataFrame with cleaned transaction data.
      contamination (float): The proportion of anomalies in the data.
      features (list): List of column names to use for anomaly detection.
    
    Returns:
      pd.DataFrame: Original DataFrame with two additional columns:
                    - 'anomaly_score': The prediction from IsolationForest.
                    - 'is_anomaly': Boolean flag where True indicates an anomaly.
    """
    if features is None:
        # Default to using 'amount_cad' if no features provided.
        features = ['amount_cad']
    
    # Ensure that provided features exist in the DataFrame
    available_features = [col for col in features if col in df.columns]
    if not available_features:
        logging.error("No valid features found for anomaly detection.")
        return df

    # Fit the IsolationForest model on the selected features
    clf = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly_score'] = clf.fit_predict(df[available_features])
    df['is_anomaly'] = df['anomaly_score'] == -1  # In IsolationForest, -1 indicates an anomaly

    logging.info(f"Anomaly detection: Found {df['is_anomaly'].sum()} anomalies out of {len(df)} rows.")
    return df

def run_anomaly_detection(input_filepath, output_filename, contamination=0.01, features=None):
    """
    Run anomaly detection on a cleaned CSV file and save the results.
    
    Parameters:
      input_filepath (str): Path to the cleaned CSV file (e.g., mnt/output/clean/cleaned_abm.csv).
      output_filename (str): Name of the file to save the anomaly detection results (e.g., anomaly_detected_abm.csv).
      contamination (float): Proportion of anomalies expected in the data.
      features (list): List of features to use for anomaly detection.
    """
    try:
        df = pd.read_csv(input_filepath)
        logging.info(f"Loaded data from {input_filepath} with {len(df)} rows.")
    except Exception as e:
        logging.error(f"Error loading {input_filepath}: {e}")
        return

    # Run anomaly detection on the dataframe
    df = detect_anomalies(df, contamination=contamination, features=features)
    
    # Save the results to the task1 output directory
    output_path = os.path.join(TASK1_OUTPUT_DIR, output_filename)
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Anomaly detection results saved to {output_path}.")
    except Exception as e:
        logging.error(f"Error saving anomaly detection results: {e}")

if __name__ == "__main__":
    # List of cleaned transaction files to process
    transaction_files = [
        "cleaned_abm.csv",
        "cleaned_card.csv",
        "cleaned_cheque.csv",
        "cleaned_eft.csv",
        "cleaned_emt.csv",
        "cleaned_wire.csv"
    ]
    
    input_dir = os.path.join("mnt", "output", "clean")
    
    # Process each file and save the anomaly detection output with a prefixed name.
    for file_name in transaction_files:
        input_file = os.path.join(input_dir, file_name)
        # Generate output file name, e.g., "anomaly_detected_abm.csv"
        output_file = "anomaly_detected_" + file_name.split("_")[1]
        run_anomaly_detection(input_file, output_file, contamination=0.01, features=['amount_cad'])
