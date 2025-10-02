import os
import json
import logging

def load_config(config_path="config.json"):
    """
    Loads configuration parameters from a JSON file.
    
    If no configuration file is found, returns an empty dictionary.
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logging.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return {}
    else:
        logging.warning(f"No config file found at {config_path}. Using default settings.")
        return {}

def save_config(config, config_path="config.json"):
    """
    Saves the configuration parameters to a JSON file.
    """
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        logging.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"Error saving config: {e}")

def ensure_directory(directory_path):
    """
    Ensures that the given directory exists. Creates it if it does not.
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        logging.info(f"Directory ensured: {directory_path}")
    except Exception as e:
        logging.error(f"Error ensuring directory {directory_path}: {e}")

def print_separator():
    """
    Prints a simple separator line to the console for clarity.
    """
    print("-" * 40)
