import os
import pandas as pd
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "yahoo_finance") # Added for loading raw data
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "yahoo_finance")
FEATURED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "featured", "yahoo_finance")

class YahooFinanceFeatureEngineer:
    """
    Engineers features for Yahoo Finance stock data, primarily technical indicators.
    Ensures the target column for LSTM ('Close' by default) remains unscaled in the output featured data.
    """
    def __init__(self, config: Dict):
        """
        Initializes the YahooFinanceFeatureEngineer.

        Args:
            config (Dict): The main configuration dictionary, expected to contain
                           a 'feature_engineering' section with 'technical_indicators'
                           and a 'models.lstm.target_column' for identifying the LSTM target.
        """
        self.config = config
        self.feature_params = config.get('feature_engineering', {}).get('technical_indicators', {})
        self.lstm_target_column = config.get('models', {}).get('lstm', {}).get('target_column', 'Close')
        if not self.feature_params:
            logging.warning("Technical indicator parameters not found in config. Missing 'feature_engineering.technical_indicators' section.")
        self._ensure_output_dirs_exist()

    def _ensure_output_dirs_exist(self) -> None:
        """Ensures that the output directories for featured data exist."""
        os.makedirs(os.path.join(FEATURED_DATA_DIR, "us_stocks"), exist_ok=True)
        os.makedirs(os.path.join(FEATURED_DATA_DIR, "indian_stocks"), exist_ok=True)
        logging.info(f"Ensured featured data directories exist at {FEATURED_DATA_DIR}")

    def _load_single_processed_data(self, ticker: str, market_type: str) -> Optional[pd.DataFrame]:
        """Loads a single processed CSV file for a given ticker and market type."""
        safe_ticker_filename = ticker.replace('.', '_').replace('-', '_') # For reading files like RELIANCE_NS_processed_data.csv
        file_path = os.path.join(PROCESSED_DATA_DIR, market_type, f"{safe_ticker_filename}_processed_data.csv")
        if not os.path.exists(file_path):
            logging.error(f"Processed data file not found for {ticker} at {file_path}")
            return None
        try:
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            logging.info(f"Successfully loaded processed data for {ticker} from {file_path}")
            return df
        except Exception as e:
            logging.error(f"Error loading processed data for {ticker} from {file_path}: {e}")
            return None

    def _load_single_raw_data(self, ticker: str, market_type: str) -> Optional[pd.DataFrame]:
        """Loads a single raw CSV file for a given ticker and market type."""
        safe_ticker_filename = ticker.replace('.', '_').replace('-', '_')
        file_path = os.path.join(RAW_DATA_DIR, market_type, f"{safe_ticker_filename}_raw_data.csv")
        if not os.path.exists(file_path):
            logging.error(f"Raw data file not found for {ticker} at {file_path}")
            return None
        try:
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            logging.info(f"Successfully loaded raw data for {ticker} from {file_path}")
            return df
        except Exception as e:
            logging.error(f"Error loading raw data for {ticker} from {file_path}: {e}")
            return None

    def _calculate_moving_averages(self, df: pd.DataFrame, column: str = 'Close') -> pd.DataFrame:
        """Calculates moving averages based on config. Uses the specified column from df."""
        ma_windows = self.feature_params.get('moving_averages', [])
        if not ma_windows:
            logging.warning(f"No moving average windows defined in config. Skipping MAs.")
            return df
        
        if column not in df.columns:
            logging.warning(f"Column '{column}' not found for MA calculation. Skipping MAs.")
            return df

        for window in ma_windows:
            df[f'MA_{window}'] = df[column].rolling(window=window).mean()
        logging.info(f"Calculated moving averages for windows: {ma_windows} using column '{column}'")
        return df

    def _calculate_rsi(self, df: pd.DataFrame, column: str = 'Close') -> pd.DataFrame:
        """Calculates Relative Strength Index (RSI) based on config. Uses the specified column from df."""
        rsi_period = self.feature_params.get('rsi_period')
        if not rsi_period:
            logging.warning(f"RSI period not defined in config. Skipping RSI.")
            return df

        if column not in df.columns:
            logging.warning(f"Column '{column}' not found for RSI calculation. Skipping RSI.")
            return df

        delta = df[column].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        
        # Add a small epsilon to prevent division by zero if loss is 0
        rs = gain / (loss + 1e-10) 
        df['RSI'] = 100 - (100 / (1 + rs))
        logging.info(f"Calculated RSI with period: {rsi_period} using column '{column}'")
        return df

    def _engineer_features_for_df(self, df_processed: pd.DataFrame, df_raw: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """
        Applies all feature engineering steps. Uses df_processed for feature calculation inputs
        and df_raw to ensure the LSTM target column is unscaled.
        """
        logging.info(f"Starting feature engineering for {ticker}...")
        
        # df_featured will start with scaled columns from df_processed for consistency in feature calculations
        df_featured = df_processed.copy()

        # Apply technical indicators using the 'Close' column from df_processed (which is scaled)
        # This is generally fine as indicators often rely on relative changes or are normalized themselves.
        # If specific indicators strictly need unscaled prices, this logic would need adjustment for those.
        indicator_input_column = 'Close' # This 'Close' is from df_processed, hence scaled.
        df_featured = self._calculate_moving_averages(df_featured, column=indicator_input_column)
        df_featured = self._calculate_rsi(df_featured, column=indicator_input_column)
        
        # Ensure the LSTM target column (e.g., 'Close') in the output df_featured is the original UNCALED version from df_raw.
        if self.lstm_target_column in df_raw.columns:
            if self.lstm_target_column in df_featured.columns:
                logging.info(f"Replacing '{self.lstm_target_column}' in featured data for {ticker} with its original unscaled values from raw data.")
                df_featured[self.lstm_target_column] = df_raw[self.lstm_target_column]
            else:
                logging.warning(f"LSTM target column '{self.lstm_target_column}' not found in processed data columns for {ticker} to replace. Adding it from raw.")
                df_featured[self.lstm_target_column] = df_raw[self.lstm_target_column]
            
            # Align indices if they somehow diverged (should not happen if processed/raw are from same source)
            df_featured = df_featured.loc[df_featured.index.isin(df_raw.index)]
            df_raw = df_raw.loc[df_raw.index.isin(df_featured.index)]
            df_featured[self.lstm_target_column] = df_raw[self.lstm_target_column].reindex(df_featured.index)

        else:
            logging.error(f"LSTM target column '{self.lstm_target_column}' not found in raw data for {ticker}. Cannot ensure it is unscaled.")
            # Depending on strictness, might return None or proceed with potentially scaled target
            return None # Safer to stop if target can't be guaranteed

        # Handle NaNs that might have been introduced by indicators or if raw target had more NaNs at start
        initial_nans = df_featured.isnull().sum().sum()
        if initial_nans > 0:
             logging.info(f"NaNs present after feature calculation/target replacement for {ticker}: {initial_nans}. Applying bfill and ffill.")
             # Important: Apply fillna carefully. Target column should reflect true past values if possible.
             # For simplicity, fill all. More advanced handling might treat target and features differently.
             df_featured.bfill(inplace=True)
             df_featured.ffill(inplace=True)
        
        final_nans = df_featured.isnull().sum().sum()
        if final_nans > 0:
            logging.warning(f"NaNs still present after fill for {ticker}: {final_nans}. Problematic columns:\n{df_featured.isnull().sum()[df_featured.isnull().sum() > 0]}")
            logging.info(f"Dropping rows with any remaining NaNs for {ticker} to ensure model compatibility.")
            df_featured.dropna(inplace=True)
            if df_featured.empty:
                logging.error(f"DataFrame for {ticker} became empty after dropping NaNs. Cannot proceed.")
                return None

        logging.info(f"Finished feature engineering for {ticker}. Shape: {df_featured.shape}. Target column '{self.lstm_target_column}' is from raw data.")
        return df_featured

    def _save_featured_data(self, featured_data: Dict[str, pd.DataFrame], market_type: str) -> None:
        """
        Saves the dataframes with engineered features to CSV files.
        """
        output_dir = os.path.join(FEATURED_DATA_DIR, market_type)
        for ticker, df in featured_data.items():
            if df is not None and not df.empty:
                safe_ticker = ticker.replace('.', '_').replace('-', '_')
                output_filename = f"{safe_ticker}_featured_data.csv"
                output_path = os.path.join(output_dir, output_filename)
                try:
                    df.to_csv(output_path)
                    logging.info(f"Successfully saved featured data for {ticker} to {output_path}")
                except Exception as e:
                    logging.error(f"Error saving featured data for {ticker}: {e}")
            elif df is None:
                 logging.warning(f"Skipped saving for {ticker} as its DataFrame was None.")
            elif df.empty:
                 logging.warning(f"Skipped saving for {ticker} as its DataFrame was empty after processing.")

    def run_feature_engineering(self) -> None:
        """
        Runs the entire feature engineering pipeline for all configured markets.
        """
        logging.info("Starting Yahoo Finance feature engineering pipeline...")
        
        yahoo_config = self.config.get('data_ingestion', {}).get('yahoo_finance', {})
        # Get a list of tickers to process for each enabled market
        markets_to_process: Dict[str, List[str]] = {}
        if yahoo_config.get('us_stocks', {}).get('enabled', False):
            markets_to_process['us_stocks'] = yahoo_config['us_stocks'].get('tickers', [])
        if yahoo_config.get('indian_stocks', {}).get('enabled', False):
            markets_to_process['indian_stocks'] = yahoo_config['indian_stocks'].get('tickers', [])

        if not markets_to_process:
            logging.warning("No markets enabled or no tickers found for Yahoo Finance feature engineering in the configuration.")
            return
        
        if not self.feature_params or (not self.feature_params.get('moving_averages') and not self.feature_params.get('rsi_period')):
            logging.warning("Feature engineering parameters for MA and RSI are not configured. Skipping feature engineering pipeline.")
            return

        for market_type, tickers_list in markets_to_process.items():
            logging.info(f"--- Engineering features for market: {market_type} ---")
            if not tickers_list:
                logging.info(f"No tickers to process for market: {market_type}")
                continue

            featured_market_data = {}
            for ticker in tickers_list:
                logging.info(f"-- Processing ticker: {ticker} in market {market_type} --")
                df_processed = self._load_single_processed_data(ticker, market_type)
                df_raw = self._load_single_raw_data(ticker, market_type)

                if df_processed is None or df_raw is None:
                    logging.warning(f"Skipping feature engineering for {ticker} due to missing processed or raw data.")
                    continue
                
                # Align dataframes by date index before proceeding to ensure they match
                # This is crucial if one df has dates the other doesn't (e.g. from different API pulls or processing)
                common_index = df_processed.index.intersection(df_raw.index)
                if common_index.empty:
                    logging.warning(f"No common dates between processed and raw data for {ticker}. Skipping.")
                    continue
                
                df_processed = df_processed.loc[common_index]
                df_raw = df_raw.loc[common_index]

                if df_processed.empty or df_raw.empty:
                    logging.warning(f"Data for {ticker} became empty after aligning processed and raw data indices. Skipping.")
                    continue

                df_featured = self._engineer_features_for_df(df_processed, df_raw, ticker)
                if df_featured is not None and not df_featured.empty:
                    featured_market_data[ticker] = df_featured
                else:
                    logging.warning(f"Feature engineering for {ticker} resulted in None or empty DataFrame. Not saving.")
            
            if featured_market_data:
                self._save_featured_data(featured_market_data, market_type)
            else:
                logging.warning(f"No data was successfully feature-engineered for market: {market_type}.")

        logging.info("Yahoo Finance feature engineering pipeline finished.")

if __name__ == '__main__':
    # This is for testing the feature engineer directly
    print("Running YahooFinanceFeatureEngineer directly for testing...")
    import sys
    PROJECT_ROOT_TEST = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, PROJECT_ROOT_TEST)
    from src.utils.config_loader import load_config # type: ignore

    test_config = load_config()
    if not test_config:
        print("Failed to load config for testing. Ensure config/config.yaml is present and valid.")
    else:
        # Ensure feature engineering and relevant data sources are enabled for testing
        if 'feature_engineering' not in test_config:
            test_config['feature_engineering'] = {}
        if 'technical_indicators' not in test_config['feature_engineering']:
            test_config['feature_engineering']['technical_indicators'] = {'moving_averages': [10, 50], 'rsi_period': 14}
        
        if 'models' not in test_config or 'lstm' not in test_config['models']:
            test_config['models'] = {'lstm': {'target_column': 'Close'}}
        elif 'target_column' not in test_config['models']['lstm']:
            test_config['models']['lstm']['target_column'] = 'Close'

        if 'data_ingestion' not in test_config or 'yahoo_finance' not in test_config['data_ingestion']:
            test_config['data_ingestion'] = {'yahoo_finance': {'us_stocks': {'enabled': True, 'tickers': ['AAPL'] }, 'indian_stocks': {'enabled': False, 'tickers': []}}}
        else:
            if 'us_stocks' not in test_config['data_ingestion']['yahoo_finance']:
                 test_config['data_ingestion']['yahoo_finance']['us_stocks'] = {'enabled': True, 'tickers': ['AAPL']}
            if 'indian_stocks' not in test_config['data_ingestion']['yahoo_finance']:
                 test_config['data_ingestion']['yahoo_finance']['indian_stocks'] = {'enabled': False, 'tickers': []}

        # For testing, ensure at least AAPL is in us_stocks for data_ingestion config
        if not test_config['data_ingestion']['yahoo_finance']['us_stocks'].get('tickers'):
            test_config['data_ingestion']['yahoo_finance']['us_stocks']['tickers'] = ['AAPL']
        elif 'AAPL' not in test_config['data_ingestion']['yahoo_finance']['us_stocks']['tickers']:
            test_config['data_ingestion']['yahoo_finance']['us_stocks']['tickers'].append('AAPL')


        print(f"Test config prepared. Target column for LSTM: {test_config['models']['lstm']['target_column']}")
        feature_engineer = YahooFinanceFeatureEngineer(config=test_config)
        feature_engineer.run_feature_engineering()
    print("Direct test run for feature engineering finished.") 