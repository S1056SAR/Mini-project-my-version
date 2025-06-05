import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler # We might need to re-scale target or ensure consistency
from typing import Tuple, List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
FEATURED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "featured", "yahoo_finance")

class StockDataset(Dataset):
    """PyTorch Dataset for stock data sequences."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class LSTMDataHandler:
    """
    Handles loading, processing, and batching of featured stock data for LSTM models.
    """
    def __init__(self, config: Dict, ticker: str, market_type: str):
        """
        Initializes the LSTMDataHandler.

        Args:
            config (Dict): The main configuration dictionary.
            ticker (str): The stock ticker symbol (e.g., 'AAPL', 'RELIANCE_NS').
            market_type (str): 'us_stocks' or 'indian_stocks'.
        """
        self.config = config
        self.ticker = ticker
        self.market_type = market_type
        self.lstm_config = config.get('models', {}).get('lstm', {})
        self.sequence_length = self.lstm_config.get('sequence_length', 60)
        self.batch_size = self.lstm_config.get('batch_size', 32)
        self.target_column = self.lstm_config.get('target_column', 'Close') # Default to 'Close'
        # Define a split ratio, e.g., 80% train, 10% validation, 10% test
        self.train_split_ratio = self.lstm_config.get('train_split_ratio', 0.8)
        self.val_split_ratio = self.lstm_config.get('val_split_ratio', 0.1)
        # Test split is implicitly 1 - train_split_ratio - val_split_ratio

        self.data_df: Optional[pd.DataFrame] = None
        self.feature_scaler = MinMaxScaler() # For features
        self.target_scaler = MinMaxScaler()  # For the target variable

    def _load_featured_data(self) -> Optional[pd.DataFrame]:
        """Loads the featured data for the specified ticker and market."""
        safe_ticker = self.ticker.replace('.', '_').replace('-', '_')
        file_path = os.path.join(FEATURED_DATA_DIR, self.market_type, f"{safe_ticker}_featured_data.csv")
        if not os.path.exists(file_path):
            logging.error(f"Featured data file not found for {self.ticker} at {file_path}")
            return None
        try:
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            logging.info(f"Successfully loaded featured data for {self.ticker} from {file_path}. Shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error loading featured data for {self.ticker}: {e}")
            return None

    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates sequences from the data.
        X will have shape (num_samples, sequence_length, num_features)
        y will have shape (num_samples, 1) (if predicting a single value)
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(target[i + self.sequence_length]) # Target is the value after the sequence
        return np.array(X), np.array(y)

    def prepare_data_loaders(self) -> Optional[Tuple[DataLoader, DataLoader, DataLoader, MinMaxScaler]]:
        """
        Loads data, preprocesses it into sequences, splits into train/val/test,
        and returns DataLoaders for each, along with the target scaler.
        """
        self.data_df = self._load_featured_data()
        if self.data_df is None or self.data_df.empty:
            logging.error(f"No data loaded for {self.ticker}, cannot prepare data loaders.")
            return None

        if self.target_column not in self.data_df.columns:
            logging.error(f"Target column '{self.target_column}' not found in data for {self.ticker}.")
            logging.info(f"Available columns: {self.data_df.columns.tolist()}")
            return None

        # Separate target series FIRST and scale it
        target_values = self.data_df[self.target_column].values.reshape(-1, 1)
        scaled_target = self.target_scaler.fit_transform(target_values)

        # For features, select only numeric columns from the dataframe
        # This excludes any non-numeric columns like a 'Ticker' string column
        features_to_scale_df = self.data_df.select_dtypes(include=np.number)
        
        if features_to_scale_df.empty:
            logging.error(f"No numeric columns found in the data for ticker {self.ticker} to use as features after type selection.")
            logging.info(f"Original columns: {self.data_df.columns.tolist()}")
            return None
        
        # Store feature names (useful for debugging or if model needs them)
        self.feature_names_ = features_to_scale_df.columns.tolist()
        logging.info(f"Using {len(self.feature_names_)} numeric features for {self.ticker}: {self.feature_names_}")

        # Scale the selected numeric features
        scaled_features_array = self.feature_scaler.fit_transform(features_to_scale_df)
        # scaled_features_array is now a numpy array. This is what _create_sequences expects for 'data'.

        # Create sequences using the scaled numeric features and the separately scaled target
        X_sequences, y_sequences = self._create_sequences(scaled_features_array, scaled_target.flatten())

        if X_sequences.shape[0] == 0:
            logging.error(f"Not enough data to create sequences for {self.ticker} with sequence length {self.sequence_length}. Data points: {len(features_to_scale_df)}")
            return None

        # Split data
        num_samples = X_sequences.shape[0]
        train_end_idx = int(num_samples * self.train_split_ratio)
        val_end_idx = train_end_idx + int(num_samples * self.val_split_ratio)

        X_train, y_train = X_sequences[:train_end_idx], y_sequences[:train_end_idx]
        X_val, y_val = X_sequences[train_end_idx:val_end_idx], y_sequences[train_end_idx:val_end_idx]
        X_test, y_test = X_sequences[val_end_idx:], y_sequences[val_end_idx:]

        logging.info(f"Data split for {self.ticker}: Train {X_train.shape[0]}, Val {X_val.shape[0]}, Test {X_test.shape[0]} samples.")

        if X_train.shape[0] == 0 or X_val.shape[0] == 0: # Test can be empty if not enough data
            logging.warning(f"Not enough data for train/validation split for {self.ticker}. Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
            # Optionally, one might decide to not proceed or use a different strategy
            # For now, we allow it but it will likely cause issues in training if val is empty

        train_dataset = StockDataset(X_train, y_train.reshape(-1,1)) # Ensure y is [samples, 1]
        val_dataset = StockDataset(X_val, y_val.reshape(-1,1))
        test_dataset = StockDataset(X_test, y_test.reshape(-1,1))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        logging.info(f"DataLoaders prepared for {self.ticker}.")
        return train_loader, val_loader, test_loader, self.target_scaler # Return scaler for inverse transform

if __name__ == '__main__':
    print("Testing LSTMDataHandler...")
    # This requires a config file and actual featured data to exist.
    # Example usage (assuming config is loaded and data exists):
    
    PROJECT_ROOT_TEST = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    sys.path.insert(0, PROJECT_ROOT_TEST) # Add project root to sys.path for util import
    import sys 
    from src.utils.config_loader import load_config # type: ignore

    test_config = load_config()
    if not test_config:
        print("Failed to load config for LSTMDataHandler test.")
    else:
        # Assuming AAPL data exists for us_stocks
        # You might need to run data ingestion, preprocessing, and feature engineering first
        example_ticker = 'AAPL' # Change to a valid ticker that has featured data
        example_market = 'us_stocks'
        
        # Ensure data source is enabled for testing
        if not test_config.get('data_ingestion', {}).get('yahoo_finance',{}).get(example_market,{}).get('enabled'):
             print(f"Yahoo finance {example_market} not enabled in config. Skipping LSTMDataHandler test for {example_ticker}.")
        elif not os.path.exists(os.path.join(FEATURED_DATA_DIR, example_market, f"{example_ticker}_featured_data.csv")):
            print(f"Featured data for {example_ticker} in {example_market} not found. Run previous pipelines. Skipping test.")
        else:
            print(f"Attempting to prepare data for {example_ticker}...")
            data_handler = LSTMDataHandler(config=test_config, ticker=example_ticker, market_type=example_market)
            loaders_output = data_handler.prepare_data_loaders()
            if loaders_output:
                train_loader, val_loader, test_loader, target_scaler = loaders_output
                print(f"Train loader has {len(train_loader)} batches.")
                if len(train_loader) > 0:
                    X_batch, y_batch = next(iter(train_loader))
                    print(f"Sample X_batch shape: {X_batch.shape}, y_batch shape: {y_batch.shape}")
                    print(f"Target scaler can be used to inverse transform predictions: {target_scaler}")
                if len(val_loader) > 0: print(f"Val loader has {len(val_loader)} batches.")
                if len(test_loader) > 0: print(f"Test loader has {len(test_loader)} batches.")
                
            else:
                print(f"Failed to prepare data loaders for {example_ticker}.")

    print("LSTMDataHandler test finished.") 