import os
import logging
import pandas as pd

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define input and output directories relative to the project root
RAW_DATA_DIR = os.path.join("mnt", "data")
CLEAN_OUTPUT_DIR = os.path.join("mnt", "output", "clean")

# Ensure the cleaned output directory exists
os.makedirs(CLEAN_OUTPUT_DIR, exist_ok=True)

def file_exists(filepath):
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return False
    return True

def clean_transactions(df, date_cols=['transaction_date'], numeric_cols=['amount_cad']):
    """
    Cleans a transaction DataFrame by:
      - Dropping duplicate rows
      - Stripping extra whitespace from string columns
      - Converting date columns to datetime
      - Combining 'transaction_date' and 'transaction_time' into a new datetime column (if both exist)
      - Converting specified numeric columns to numeric types
      - Standardizing 'debit_credit' to lowercase
      - Standardizing geographic columns ('country', 'province', 'city') to uppercase
      - Dropping rows missing 'customer_id'
    """
    initial_rows = len(df)
    df = df.copy()

    try:
        df.drop_duplicates(inplace=True)
    except Exception as e:
        logging.error(f"Error dropping duplicates: {e}")

    try:
        # Strip extra whitespace from all object-type columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
    except Exception as e:
        logging.error(f"Error stripping whitespace: {e}")

    try:
        # Convert date columns to datetime
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    except Exception as e:
        logging.error(f"Error converting date columns: {e}")

    try:
        # Combine 'transaction_date' and 'transaction_time' if available
        if 'transaction_date' in df.columns and 'transaction_time' in df.columns:
            df['transaction_time'] = df['transaction_time'].fillna("00:00:00")
            df['transaction_datetime'] = pd.to_datetime(
                df['transaction_date'].dt.strftime('%Y-%m-%d') + ' ' + df['transaction_time'],
                errors='coerce'
            )
    except Exception as e:
        logging.error(f"Error combining date and time: {e}")

    try:
        # Convert specified numeric columns to numeric types
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception as e:
        logging.error(f"Error converting numeric columns: {e}")

    try:
        # Standardize the 'debit_credit' column to lowercase if present
        if 'debit_credit' in df.columns:
            df['debit_credit'] = df['debit_credit'].str.lower()
    except Exception as e:
        logging.error(f"Error standardizing 'debit_credit': {e}")

    try:
        # Standardize geographic columns to uppercase
        for col in ['country', 'province', 'city']:
            if col in df.columns:
                df[col] = df[col].str.upper()
    except Exception as e:
        logging.error(f"Error standardizing geographic columns: {e}")

    try:
        # Drop rows missing 'customer_id'
        if 'customer_id' in df.columns:
            df.dropna(subset=['customer_id'], inplace=True)
    except Exception as e:
        logging.error(f"Error dropping rows missing 'customer_id': {e}")

    final_rows = len(df)
    logging.info(f"Transactions cleaned: {initial_rows} -> {final_rows} rows")
    return df

def clean_bool_columns(df, bool_columns):
    """
    Convert specified columns to boolean types.
    """
    for col in bool_columns:
        if col in df.columns:
            try:
                df[col] = df[col].astype(bool)
            except Exception as e:
                logging.error(f"Error converting {col} to bool: {e}")
    return df

def clean_kyc_data(df):
    """
    Cleans a KYC DataFrame by:
      - Resetting the index so that 'customer_id' becomes a column
      - Dropping duplicate rows
      - Stripping extra whitespace from string columns
      - Converting date columns to datetime
      - Converting numeric columns (e.g., 'sales', 'employee_count') to numeric types
      - Standardizing geographic and industry code columns to uppercase
      - Dropping rows missing 'customer_id'
    """
    initial_rows = len(df)
    df = df.copy()
    
    try:
        df.reset_index(inplace=True)
        df.drop_duplicates(inplace=True)
    except Exception as e:
        logging.error(f"Error resetting index or dropping duplicates in KYC: {e}")

    try:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
    except Exception as e:
        logging.error(f"Error stripping whitespace in KYC data: {e}")

    try:
        if 'established_date' in df.columns:
            df['established_date'] = pd.to_datetime(df['established_date'], errors='coerce')
        if 'onboard_date' in df.columns:
            df['onboard_date'] = pd.to_datetime(df['onboard_date'], errors='coerce')
    except Exception as e:
        logging.error(f"Error converting date columns in KYC data: {e}")

    try:
        if 'sales' in df.columns:
            df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        if 'employee_count' in df.columns:
            df['employee_count'] = pd.to_numeric(df['employee_count'], errors='coerce')
    except Exception as e:
        logging.error(f"Error converting numeric columns in KYC data: {e}")

    try:
        for col in ['country', 'province', 'city', 'industry_code']:
            if col in df.columns:
                df[col] = df[col].str.upper()
    except Exception as e:
        logging.error(f"Error standardizing columns in KYC data: {e}")

    try:
        df.dropna(subset=['customer_id'], inplace=True)
    except Exception as e:
        logging.error(f"Error dropping rows with missing customer_id in KYC: {e}")

    final_rows = len(df)
    logging.info(f"KYC data cleaned: {initial_rows} -> {final_rows} rows")
    return df

def clean_kyc_industry_codes(df):
    """
    Cleans the KYC industry codes data by:
      - Dropping duplicates
      - Resetting the index so that 'industry_code' becomes a column
      - Stripping whitespace and standardizing to uppercase for all object columns
    """
    try:
        df = df.copy()
        df.drop_duplicates(inplace=True)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'industry_code'}, inplace=True)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip().str.upper()
    except Exception as e:
        logging.error(f"Error cleaning KYC industry codes data: {e}")
    return df

def save_cleaned_data(df, filename):
    """
    Saves the given DataFrame to the CLEAN_OUTPUT_DIR with the provided filename.
    """
    output_path = os.path.join(CLEAN_OUTPUT_DIR, filename)
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Saved cleaned data to {output_path}")
    except Exception as e:
        logging.error(f"Error saving file {filename}: {e}")

if __name__ == "__main__":
    # List of raw data files to load for transactions
    transaction_files = {
        "abm.csv": None,
        "card.csv": None,
        "cheque.csv": None,
        "eft.csv": None,
        "emt.csv": None,
        "wire.csv": None
    }

    # Load each transaction file if it exists
    for file_name in transaction_files.keys():
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        if file_exists(file_path):
            try:
                transaction_files[file_name] = pd.read_csv(file_path, index_col=0)
                logging.info(f"Loaded {file_name}")
            except Exception as e:
                logging.error(f"Error loading {file_name}: {e}")
                raise
        else:
            raise FileNotFoundError(f"{file_path} not found.")

    # Load raw KYC and industry codes data
    kyc_file = os.path.join(RAW_DATA_DIR, "kyc.csv")
    kyc_industry_codes_file = os.path.join(RAW_DATA_DIR, "kyc_industry_codes.csv")
    if not file_exists(kyc_file) or not file_exists(kyc_industry_codes_file):
        raise FileNotFoundError("KYC data files not found in the expected directory.")
    try:
        kyc_data = pd.read_csv(kyc_file, index_col=0)
        kyc_industry_codes = pd.read_csv(kyc_industry_codes_file, index_col=0)
        logging.info("Loaded KYC data files.")
    except Exception as e:
        logging.error(f"Error loading KYC data: {e}")
        raise

    # Clean transaction data
    abm_data = clean_transactions(transaction_files["abm.csv"], date_cols=['transaction_date'], numeric_cols=['amount_cad'])
    card_data = clean_transactions(transaction_files["card.csv"], date_cols=['transaction_date'], numeric_cols=['amount_cad'])
    cheque_data = clean_transactions(transaction_files["cheque.csv"], date_cols=['transaction_date'], numeric_cols=['amount_cad'])
    eft_data = clean_transactions(transaction_files["eft.csv"], date_cols=['transaction_date'], numeric_cols=['amount_cad'])
    emt_data = clean_transactions(transaction_files["emt.csv"], date_cols=['transaction_date'], numeric_cols=['amount_cad'])
    wire_data = clean_transactions(transaction_files["wire.csv"], date_cols=['transaction_date'], numeric_cols=['amount_cad'])

    # Additional enhancement: For card_data, convert negative amounts to absolute values
    if 'amount_cad' in card_data.columns:
        card_data['amount_cad'] = card_data['amount_cad'].abs()
        logging.info("Converted negative amounts in card_data to absolute values.")

    # Convert known boolean columns explicitly
    abm_data = clean_bool_columns(abm_data, ['cash_indicator'])
    card_data = clean_bool_columns(card_data, ['ecommerce_ind'])

    # Clean KYC data and industry codes
    kyc_data = clean_kyc_data(kyc_data)
    kyc_industry_codes = clean_kyc_industry_codes(kyc_industry_codes)

    # Save cleaned transaction data
    save_cleaned_data(abm_data, "cleaned_abm.csv")
    save_cleaned_data(card_data, "cleaned_card.csv")
    save_cleaned_data(cheque_data, "cleaned_cheque.csv")
    save_cleaned_data(eft_data, "cleaned_eft.csv")
    save_cleaned_data(emt_data, "cleaned_emt.csv")
    save_cleaned_data(wire_data, "cleaned_wire.csv")

    # Save cleaned KYC data and industry codes
    save_cleaned_data(kyc_data, "cleaned_kyc.csv")
    save_cleaned_data(kyc_industry_codes, "cleaned_kyc_industry_codes.csv")

    logging.info("Data cleaning complete. Please review the logs for any issues.")
