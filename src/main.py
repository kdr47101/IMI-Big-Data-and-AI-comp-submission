import os
import logging
from anomaly_detection import run_anomaly_detection
from embeddings import generate_customer_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting production pipeline...")

    # Define the list of cleaned transaction files (Task 1)
    transaction_files = [
        "cleaned_abm.csv",
        "cleaned_card.csv",
        "cleaned_cheque.csv",
        "cleaned_eft.csv",
        "cleaned_emt.csv",
        "cleaned_wire.csv"
    ]
    clean_dir = os.path.join("mnt", "output", "clean")
    
    # Run anomaly detection on each cleaned transaction file
    for file_name in transaction_files:
        input_path = os.path.join(clean_dir, file_name)
        # Create an output filename by replacing 'cleaned_' with 'anomaly_detected_'
        base = file_name.replace("cleaned_", "")
        output_filename = f"anomaly_detected_{base}"
        logging.info(f"Running anomaly detection on {input_path}")
        run_anomaly_detection(input_path, output_filename, contamination=0.01, features=['amount_cad'])
    
    # Generate customer embeddings (Task 3)
    logging.info("Generating customer embeddings from KYC data...")
    # Adjust feature columns as needed (e.g., 'employee_count' and 'sales' are used here)
    generate_customer_embeddings(feature_cols=['employee_count', 'sales'], epochs=20, batch_size=32)
    
    logging.info("Production pipeline complete.")

if __name__ == "__main__":
    main()
