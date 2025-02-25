import os
import pickle
import joblib
import tensorflow as tf
import logging
import pandas as pd

from config import MODEL_DIR, STOCK_LIST_FILE

# Configure logging
logging.basicConfig(level=logging.INFO)  # Change to ERROR to suppress logs
logger = logging.getLogger(__name__)

def preprocess_input(X_input, feature_scaler):
    """Ensure input data has correct feature names before scaling."""
    try:
        if hasattr(feature_scaler, "feature_names_in_"):
            expected_features = feature_scaler.feature_names_in_
            if len(X_input.shape) == 1:  # Convert 1D array to DataFrame
                X_input = pd.DataFrame([X_input], columns=expected_features)
            else:
                X_input = pd.DataFrame(X_input, columns=expected_features)

        return feature_scaler.transform(X_input)
    
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return None  # Return None to indicate failure

def load_stock_names():
    """Load the list of available trained stocks safely."""
    if not os.path.exists(STOCK_LIST_FILE):
        logger.warning(f"Stock list file '{STOCK_LIST_FILE}' not found. Returning an empty list.")
        return []  # Return an empty list if the file is missing
    
    try:
        with open(STOCK_LIST_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading stock list file: {e}")
        return []  # Return an empty list if loading fails

def load_model(stock_symbol):
    """Load the LSTM model and scalers for a specific stock."""
    model_path = os.path.join(MODEL_DIR, f"{stock_symbol}.h5")
    feature_scaler_path = os.path.join(MODEL_DIR, f"{stock_symbol}_feature_scaler.pkl")
    target_scaler_path = os.path.join(MODEL_DIR, f"{stock_symbol}_target_scaler.pkl")

    # ✅ Check if model files exist
    missing_files = []
    for path in [model_path, feature_scaler_path, target_scaler_path]:
        if not os.path.exists(path):
            missing_files.append(path)

    if missing_files:
        logger.warning(f"Missing model files for {stock_symbol}: {', '.join(missing_files)}")
        return None, None, None

    try:
        # ✅ Load Model
        model = tf.keras.models.load_model(model_path)

        # ✅ Load Scalers
        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)

        return model, feature_scaler, target_scaler

    except Exception as e:
        logger.error(f"Error loading model for {stock_symbol}: {e}")
        return None, None, None
