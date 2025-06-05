import os
import torch
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler

from .lstm_model import LSTMModel
from .lstm_data_handler import LSTMDataHandler # To get test data and input_features
from src.utils.config_loader import load_config # For standalone testing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODELS_LOAD_DIR = os.path.join(PROJECT_ROOT, "experiments", "lstm_models")
PREDICTIONS_SAVE_DIR = os.path.join(PROJECT_ROOT, "experiments", "lstm_predictions")

class LSTMPredictor:
    """
    Handles loading a trained LSTM model and making predictions on test data.
    """
    def __init__(self, config: Dict, ticker: str, market_type: str):
        """
        Initializes the LSTMPredictor.

        Args:
            config (Dict): The main configuration dictionary.
            ticker (str): The stock ticker symbol (e.g., 'AAPL', 'RELIANCE_NS').
            market_type (str): 'us_stocks' or 'indian_stocks'.
        """
        self.config = config
        self.ticker = ticker
        self.market_type = market_type
        self.lstm_global_config = config.get('models', {}).get('lstm', {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model_name = "best_model.pth"
        self.scaler_name = "target_scaler.pkl"
        
        self.model_load_path = os.path.join(MODELS_LOAD_DIR, self.market_type, self.ticker)
        self.model_file_path = os.path.join(self.model_load_path, self.model_name)
        self.scaler_file_path = os.path.join(self.model_load_path, self.scaler_name)

        self.predictions_save_path = os.path.join(PREDICTIONS_SAVE_DIR, self.market_type, self.ticker)
        os.makedirs(self.predictions_save_path, exist_ok=True)

        self.model: Optional[LSTMModel] = None
        self.target_scaler: Optional[MinMaxScaler] = None
        self.data_handler: Optional[LSTMDataHandler] = None
        self.test_loader: Optional[torch.utils.data.DataLoader] = None


    def _load_model_and_scaler(self) -> bool:
        """Loads the trained model and target scaler."""
        if not os.path.exists(self.model_file_path):
            logging.error(f"Model file not found at {self.model_file_path}")
            return False
        if not os.path.exists(self.scaler_file_path):
            logging.error(f"Scaler file not found at {self.scaler_file_path}")
            return False

        try:
            self.target_scaler = joblib.load(self.scaler_file_path)
            logging.info(f"Target scaler loaded successfully from {self.scaler_file_path}")
        except Exception as e:
            logging.error(f"Error loading target scaler from {self.scaler_file_path}: {e}")
            return False

        # We need to instantiate DataHandler to get test_loader and input_features count
        # The feature_scaler within data_handler will be fit on the specific ticker's data, which is fine.
        # The target_scaler we use for inverse transform is the one loaded from file.
        self.data_handler = LSTMDataHandler(config=self.config, ticker=self.ticker, market_type=self.market_type)
        
        # We need a way to get input_features without necessarily going through the full train/val/test split if possible,
        # or at least get it from the test data. The data_handler.prepare_data_loaders() also returns the test_loader.
        # Let's get X_test's shape to determine input_features.
        
        # Temporarily get all loaders to find input_features from test data structure
        # Note: This will also fit internal scalers in data_handler, but we use our loaded target_scaler
        all_loaders_output = self.data_handler.prepare_data_loaders()
        if not all_loaders_output:
            logging.error(f"Failed to prepare data loaders via LSTMDataHandler for {self.ticker} to determine input features.")
            return False
        
        _, _, self.test_loader, _ = all_loaders_output # We only need test_loader here; ignore its scaler

        if not self.test_loader or len(self.test_loader) == 0:
            logging.error(f"Test loader for {self.ticker} is empty or None. Cannot proceed with prediction.")
            return False
            
        try:
            X_sample_batch, _ = next(iter(self.test_loader))
            input_features = X_sample_batch.shape[2] # (batch, seq_len, features)
            logging.info(f"Determined input_features: {input_features} for {self.ticker} from test data.")
        except StopIteration:
            logging.error(f"Test loader for {self.ticker} is empty, cannot determine input features for model loading.")
            return False
        except Exception as e:
            logging.error(f"Error getting sample batch from test_loader for {self.ticker} to determine input features: {e}")
            return False

        # Initialize model architecture
        model_params_from_config = self.lstm_global_config.copy()
        self.model = LSTMModel(input_features=input_features, model_config=model_params_from_config).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(torch.load(self.model_file_path, map_location=self.device))
        self.model.eval() # Set model to evaluation mode
        logging.info(f"Model loaded successfully from {self.model_file_path} and set to eval mode.")
        return True

    def predict(self) -> Optional[pd.DataFrame]:
        """
        Makes predictions on the test set and saves them.
        Returns a DataFrame with original scale predictions and actuals.
        """
        if not self._load_model_and_scaler():
            logging.error(f"Could not load model and/or scaler for {self.ticker}. Aborting prediction.")
            return None

        all_predictions_scaled = []
        all_actuals_scaled = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                # y_batch is already on CPU from DataLoader, but we'll move it for consistency if needed or for loss calc
                
                output_scaled = self.model(X_batch)
                
                all_predictions_scaled.append(output_scaled.cpu().numpy())
                all_actuals_scaled.append(y_batch.cpu().numpy()) # y_batch is (batch_size, 1) from StockDataset

        if not all_predictions_scaled:
            logging.warning(f"No predictions were generated for {self.ticker}.")
            return None

        predictions_scaled_np = np.concatenate(all_predictions_scaled)
        actuals_scaled_np = np.concatenate(all_actuals_scaled)

        # Inverse transform using the loaded scaler
        predictions_original_scale = self.target_scaler.inverse_transform(predictions_scaled_np)
        actuals_original_scale = self.target_scaler.inverse_transform(actuals_scaled_np)
        
        # Get the corresponding dates from the original dataframe used by data_handler
        # The test data corresponds to the latter part of data_df in LSTMDataHandler
        # after sequence creation and splitting.
        
        # We need to reconstruct the dates for the test predictions.
        # The number of predictions is len(actuals_original_scale)
        # The test data started at val_end_idx in LSTMDataHandler._create_sequences output
        # The original dataframe is self.data_handler.data_df
        
        original_df_for_dates = self.data_handler.data_df
        if original_df_for_dates is None:
            logging.error("Original dataframe (data_df) not found in data_handler. Cannot align dates.")
            return None

        # Calculate split indices as in LSTMDataHandler to find where test data begins
        # This is a bit of re-implementation of logic from data_handler, ideally date alignment is more robust
        num_total_samples_before_split = len(original_df_for_dates) - self.lstm_global_config.get('sequence_length', 60)
        if num_total_samples_before_split <= 0:
             logging.error("Not enough data in original_df_for_dates to form any sequences.")
             return None

        train_ratio = self.lstm_global_config.get('train_split_ratio', 0.8)
        val_ratio = self.lstm_global_config.get('val_split_ratio', 0.1)
        
        train_end_idx_abs = int(num_total_samples_before_split * train_ratio)
        val_end_idx_abs = train_end_idx_abs + int(num_total_samples_before_split * val_ratio)
        
        # The y_sequences (and thus our test actuals/predictions) start from sequence_length-th original data point
        # and then are indexed by train_end_idx_abs, val_end_idx_abs.
        # So the first test actual corresponds to original data at index: sequence_length + val_end_idx_abs
        first_test_actual_original_df_index = self.lstm_global_config.get('sequence_length', 60) + val_end_idx_abs
        
        if first_test_actual_original_df_index >= len(original_df_for_dates.index):
            logging.error(f"Calculated start index for test dates ({first_test_actual_original_df_index}) is out of bounds for original data of length {len(original_df_for_dates.index)}.")
            # This can happen if test set is very small or empty.
            # test_loader check should catch empty test sets, but this is a safeguard.
            if len(actuals_original_scale) == 0: # If there truly are no test samples, this is expected
                 logging.info("No test samples to align dates for, likely an empty test set.")
                 return pd.DataFrame({'Date': [], 'Actual': [], 'Predicted': []}) # Return empty df
            return None


        num_predictions = len(actuals_original_scale)
        test_dates = original_df_for_dates.index[first_test_actual_original_df_index : first_test_actual_original_df_index + num_predictions]
        
        if len(test_dates) != num_predictions:
            logging.warning(f"Mismatch in length of test dates ({len(test_dates)}) and number of predictions ({num_predictions}). Date alignment might be incorrect.")
            # Fallback or error, for now, we'll proceed but this needs attention if it occurs.
            # Ensure test_dates is not longer than predictions to avoid index errors.
            test_dates = test_dates[:num_predictions]


        results_df = pd.DataFrame({
            'Date': test_dates,
            'Actual': actuals_original_scale.flatten(),
            'Predicted': predictions_original_scale.flatten()
        })
        results_df.set_index('Date', inplace=True)

        # Save predictions
        pred_filename = os.path.join(self.predictions_save_path, f"{self.ticker}_predictions.csv")
        try:
            results_df.to_csv(pred_filename)
            logging.info(f"Predictions saved to {pred_filename}")
        except Exception as e:
            logging.error(f"Error saving predictions to {pred_filename}: {e}")
        
        return results_df

if __name__ == '__main__':
    print("Testing LSTMPredictor...")
    # This requires a config file, trained model, and its scaler to exist.
    # Example usage:
    
    test_config = load_config()
    if not test_config:
        print("Failed to load config for LSTMPredictor test. Ensure config/config.yaml is present.")
    else:
        # Make sure AAPL model and scaler exist from a previous training run that saved the scaler
        example_ticker = 'AAPL' 
        example_market = 'us_stocks'

        model_file = os.path.join(MODELS_LOAD_DIR, example_market, example_ticker, "best_model.pth")
        scaler_file = os.path.join(MODELS_LOAD_DIR, example_market, example_ticker, "target_scaler.pkl")
        featured_file = os.path.join(PROJECT_ROOT, "data", "featured", "yahoo_finance", example_market, f"{example_ticker}_featured_data.csv")

        if not os.path.exists(model_file) or not os.path.exists(scaler_file):
            print(f"Model ({model_file}) or scaler ({scaler_file}) not found for {example_ticker}. Run training first (ensure it saves scaler). Skipping predictor test.")
        elif not os.path.exists(featured_file):
            print(f"Featured data for {example_ticker} not found at {featured_file}. Run feature engineering. Skipping predictor test.")
        else:
            print(f"Attempting to run predictions for {example_ticker}...")
            predictor = LSTMPredictor(config=test_config, ticker=example_ticker, market_type=example_market)
            predictions_df = predictor.predict()
            if predictions_df is not None:
                print(f"Predictions generated for {example_ticker}:")
                print(predictions_df.head())
                # Check if file was saved
                pred_output_file = os.path.join(PREDICTIONS_SAVE_DIR, example_market, example_ticker, f"{example_ticker}_predictions.csv")
                if os.path.exists(pred_output_file):
                    print(f"Predictions successfully saved to {pred_output_file}")
                else:
                    print(f"Error: Prediction file was not saved to {pred_output_file}")
            else:
                print(f"Failed to generate predictions for {example_ticker}.")
    print("LSTMPredictor test finished.") 