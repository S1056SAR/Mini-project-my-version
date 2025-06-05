import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "yahoo_finance")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "yahoo_finance")

class YahooFinancePreprocessor:
    """
    Preprocesses raw Yahoo Finance stock data.
    Handles loading, cleaning, normalization, and saving of data.
    """
    def __init__(self, config: Dict):
        """
        Initializes the YahooFinancePreprocessor.

        Args:
            config (Dict): The main configuration dictionary.
        """
        self.config = config
        self.scaler = MinMaxScaler() # Or StandardScaler, as per preference/rules
        self._ensure_output_dirs_exist()

    def _ensure_output_dirs_exist(self) -> None:
        """Ensures that the output directories for processed data exist."""
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, "us_stocks"), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, "indian_stocks"), exist_ok=True)
        logging.info(f"Ensured processed data directories exist at {PROCESSED_DATA_DIR}")

    def _load_raw_data_for_market(self, market_type: str) -> Dict[str, pd.DataFrame]:
        """
        Loads all raw CSV files for a given market type (e.g., 'us_stocks', 'indian_stocks').

        Args:
            market_type (str): The market type ('us_stocks' or 'indian_stocks').

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where keys are ticker symbols (or filenames)
                                     and values are pandas DataFrames.
        """
        market_data_path = os.path.join(RAW_DATA_DIR, market_type)
        all_data = {}
        if not os.path.exists(market_data_path):
            logging.warning(f"Raw data directory not found for market {market_type} at {market_data_path}")
            return all_data

        for filename in os.listdir(market_data_path):
            if filename.endswith("_raw_data.csv"):
                file_path = os.path.join(market_data_path, filename)
                try:
                    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                    # Extract ticker from filename (e.g., AAPL_raw_data.csv -> AAPL)
                    ticker = filename.replace("_raw_data.csv", "")
                    all_data[ticker] = df
                    logging.info(f"Successfully loaded raw data for {ticker} from {file_path}")
                except Exception as e:
                    logging.error(f"Error loading raw data for {filename}: {e}")
        return all_data

    def _preprocess_single_df(self, df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """
        Preprocesses a single stock's DataFrame.

        Args:
            df (pd.DataFrame): The raw DataFrame for a single stock.
            ticker (str): The ticker symbol for logging.

        Returns:
            Optional[pd.DataFrame]: The preprocessed DataFrame, or None if processing fails.
        """
        logging.info(f"Starting preprocessing for {ticker}...")
        
        # 1. Handle Missing Values (using forward fill, then backward fill)
        # Check initial missing values
        initial_missing = df.isnull().sum().sum()
        if initial_missing > 0:
            logging.info(f"Missing values before fill for {ticker}: \n{df.isnull().sum()}")
        
        # Columns to apply ffill/bfill (typically OHLCV)
        cols_to_fill = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in cols_to_fill:
            if col not in df.columns:
                logging.warning(f"Column {col} not found in DataFrame for {ticker}. Skipping fill for this column.")
                continue
            df[col] = df[col].ffill()
            df[col] = df[col].bfill()

        # Check missing values after fill
        final_missing = df.isnull().sum().sum()
        if final_missing > 0:
            logging.warning(f"Remaining missing values after ffill/bfill for {ticker}: \n{df.isnull().sum()}")
            # Option: drop rows with any remaining NaNs if critical
            # df.dropna(inplace=True)
            # logging.info(f"Dropped rows with remaining NaNs for {ticker}.")
        
        # 2. Ensure Data Types (already handled by parse_dates in load, numeric by yfinance)
        # We can add explicit checks if needed. For example:
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN, then handle
        
        # Re-check for NaNs after numeric conversion, especially if 'coerce' created new ones
        if df.isnull().sum().sum() > final_missing: # if new NaNs appeared
             logging.warning(f"NaNs appeared after numeric conversion for {ticker}. Re-filling...")
             for col in cols_to_fill:
                 if col in df.columns:
                    df[col] = df[col].ffill().bfill() # Re-apply fill
             if df.isnull().sum().sum() > 0:
                logging.error(f"Could not resolve all NaNs for {ticker} after numeric conversion and re-fill. Skipping this stock.")
                return None


        # 3. Normalization/Scaling (OHLCV columns)
        # We scale 'Open', 'High', 'Low', 'Close', 'Volume'. 'Adj Close' is often used as target, 
        # but scaling it depends on the modeling strategy. For now, let's scale it too.
        cols_to_scale = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        # Filter out columns that might not exist (e.g. if a stock has no volume data)
        scalable_cols_present = [col for col in cols_to_scale if col in df.columns and df[col].notna().all()]

        if not scalable_cols_present:
            logging.warning(f"No columns to scale or columns contain NaNs for {ticker}. Skipping scaling.")
        else:
            # Fit scaler on the first stock's data for each market, then transform others.
            # For simplicity here, we fit and transform per stock.
            # A better approach for consistency might be to fit on concatenated data or a reference period.
            # However, individual scaling is common for individual stock models.
            # Rule: "For LSTM models, use PyTorch's nn.LSTM modules." - scaling is generally good.
            df_scaled = df.copy()
            try:
                df_scaled[scalable_cols_present] = self.scaler.fit_transform(df[scalable_cols_present])
                # Check for NaNs post-scaling (should not happen with MinMaxScaler if no NaNs prior)
                if df_scaled[scalable_cols_present].isnull().sum().any():
                    logging.error(f"NaNs introduced during scaling for {ticker}. This is unexpected.")
                    # Fallback to original data or handle appropriately
                    return df # Or None, or df without scaling
                df = df_scaled
                logging.info(f"Successfully scaled data for {ticker}.")
            except ValueError as e:
                logging.error(f"ValueError during scaling for {ticker}: {e}. Columns: {scalable_cols_present}. Data sample:\n{df[scalable_cols_present].head()}")
                # This can happen if a column is all NaNs or has non-numeric types not caught before.
                return None # Skip this stock if scaling fails catastrophically


        # 4. Drop original columns if new scaled ones are preferred and named differently (not the case here)
        # 5. Add any other preprocessing steps here (e.g., feature engineering if not a separate module)

        logging.info(f"Finished preprocessing for {ticker}. Shape: {df.shape}")
        return df

    def _save_processed_data(self, processed_data: Dict[str, pd.DataFrame], market_type: str) -> None:
        """
        Saves the processed dataframes to CSV files.

        Args:
            processed_data (Dict[str, pd.DataFrame]): Dictionary of processed dataframes.
            market_type (str): The market type ('us_stocks' or 'indian_stocks').
        """
        output_dir = os.path.join(PROCESSED_DATA_DIR, market_type)
        for ticker, df in processed_data.items():
            if df is not None:
                # Sanitize ticker for filename (relevant for Indian stocks like 'RELIANCE.NS')
                safe_ticker = ticker.replace('.', '_').replace('-', '_')
                output_filename = f"{safe_ticker}_processed_data.csv"
                output_path = os.path.join(output_dir, output_filename)
                try:
                    df.to_csv(output_path)
                    logging.info(f"Successfully saved processed data for {ticker} to {output_path}")
                except Exception as e:
                    logging.error(f"Error saving processed data for {ticker}: {e}")
            else:
                logging.warning(f"Skipped saving for {ticker} as its DataFrame was None (likely due to processing errors).")


    def run_preprocessing(self) -> None:
        """
        Runs the entire preprocessing pipeline for all configured markets.
        """
        logging.info("Starting Yahoo Finance data preprocessing pipeline...")
        
        yahoo_config = self.config.get('data_ingestion', {}).get('yahoo_finance', {})
        market_types = []
        if yahoo_config.get('us_stocks', {}).get('enabled', False):
            market_types.append('us_stocks')
        if yahoo_config.get('indian_stocks', {}).get('enabled', False):
            market_types.append('indian_stocks')

        if not market_types:
            logging.warning("No markets enabled for Yahoo Finance preprocessing in the configuration.")
            return

        for market in market_types:
            logging.info(f"--- Processing market: {market} ---")
            raw_market_data = self._load_raw_data_for_market(market)
            if not raw_market_data:
                logging.warning(f"No raw data loaded for market: {market}. Skipping.")
                continue
            
            processed_market_data = {}
            for ticker, df_raw in raw_market_data.items():
                df_processed = self._preprocess_single_df(df_raw.copy(), ticker) # Pass a copy
                if df_processed is not None:
                    processed_market_data[ticker] = df_processed
            
            if processed_market_data:
                self._save_processed_data(processed_market_data, market)
            else:
                logging.warning(f"No data was successfully processed for market: {market}.")

        logging.info("Yahoo Finance data preprocessing pipeline finished.")

if __name__ == '__main__':
    # This is for testing the preprocessor directly
    # You would typically run this via main.py
    print("Running YahooFinancePreprocessor directly for testing...")
    
    # Construct a dummy config for direct execution
    # In a real scenario, this comes from config_loader via main.py
    PROJECT_ROOT_MAIN_TEST = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, PROJECT_ROOT_MAIN_TEST)
    from src.utils.config_loader import load_config #type: ignore

    test_config = load_config()
    if not test_config:
        print("Failed to load config for testing. Ensure config/config.yaml is present and valid.")
    else:
        print(f"Test Config loaded. Version: {test_config.get('version', 'N/A')}")
        # Ensure Yahoo Finance source is enabled for testing
        if not test_config.get('data_ingestion', {}).get('yahoo_finance',{}):
             test_config['data_ingestion'] = {'yahoo_finance': {'us_stocks': {'enabled': True}, 'indian_stocks': {'enabled': True}}}
        elif not test_config['data_ingestion']['yahoo_finance'].get('us_stocks'):
             test_config['data_ingestion']['yahoo_finance']['us_stocks'] = {'enabled': True}
        elif not test_config['data_ingestion']['yahoo_finance'].get('indian_stocks'):
             test_config['data_ingestion']['yahoo_finance']['indian_stocks'] = {'enabled': True}

        preprocessor = YahooFinancePreprocessor(config=test_config)
        preprocessor.run_preprocessing()
    print("Direct test run finished.") 