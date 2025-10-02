import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories relative to the project root
CLEAN_DATA_DIR = os.path.join("mnt", "output", "clean")
TASK3_OUTPUT_DIR = os.path.join("mnt", "output", "task3")
os.makedirs(TASK3_OUTPUT_DIR, exist_ok=True)

def load_kyc_data():
    """Load the cleaned KYC data from mnt/output/clean/cleaned_kyc.csv."""
    kyc_file = os.path.join(CLEAN_DATA_DIR, "cleaned_kyc.csv")
    try:
        df = pd.read_csv(kyc_file)
        logging.info(f"Loaded KYC data with {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"Error loading KYC data: {e}")
        return None

def preprocess_kyc_data(df, feature_cols):
    """
    Selects the given feature columns from the KYC DataFrame,
    fills missing values, and applies simple min-max normalization.
    """
    X = df[feature_cols].fillna(0)
    # Simple min-max scaling
    X_norm = (X - X.min()) / (X.max() - X.min() + 1e-6)
    return X_norm.values

def build_autoencoder(input_dim, embedding_dim=16):
    """
    Builds a simple autoencoder with one hidden encoding layer.
    
    Parameters:
      input_dim (int): Number of input features.
      embedding_dim (int): Dimension of the latent embedding.
    
    Returns:
      autoencoder (Model): Compiled autoencoder model.
      encoder (Model): Model to extract embeddings.
    """
    input_layer = Input(shape=(input_dim,))
    # Encoder
    encoded = Dense(32, activation='relu')(input_layer)
    encoded = Dense(embedding_dim, activation='relu')(encoded)
    # Decoder
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def generate_customer_embeddings(feature_cols, epochs=20, batch_size=32):
    """
    Generates customer embeddings using an autoencoder trained on selected features.
    
    Parameters:
      feature_cols (list): List of column names from KYC data to use as features.
      epochs (int): Number of epochs to train the autoencoder.
      batch_size (int): Batch size for training.
    """
    # Load KYC data
    df = load_kyc_data()
    if df is None:
        logging.error("Failed to load KYC data. Exiting embedding generation.")
        return
    
    # Preprocess data and prepare features
    X = preprocess_kyc_data(df, feature_cols)
    input_dim = X.shape[1]
    logging.info(f"Building autoencoder with input dimension {input_dim} and embedding dimension 16.")
    
    # Build and train the autoencoder
    autoencoder, encoder = build_autoencoder(input_dim, embedding_dim=16)
    logging.info("Training autoencoder...")
    autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Generate embeddings using the encoder model
    embeddings = encoder.predict(X)
    
    # Prepare output: each line starts with customer_id followed by the embedding vector values
    output_lines = []
    customer_ids = df['customer_id'].astype(str).values
    for cid, emb in zip(customer_ids, embeddings):
        emb_str = ", ".join(f"{x:.8f}" for x in emb)
        output_lines.append(f"{cid}, {emb_str}")
    
    output_file = os.path.join(TASK3_OUTPUT_DIR, "customer_embeddings.txt")
    try:
        with open(output_file, "w") as f:
            for line in output_lines:
                f.write(line + "\n")
        logging.info(f"Customer embeddings saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving customer embeddings: {e}")

if __name__ == "__main__":
    # Specify which columns to use for generating embeddings.
    # Adjust these based on your cleaned KYC data. For example, you might use numeric features like 'employee_count' and 'sales'.
    feature_columns = ['employee_count', 'sales']
    generate_customer_embeddings(feature_columns, epochs=20, batch_size=32)
