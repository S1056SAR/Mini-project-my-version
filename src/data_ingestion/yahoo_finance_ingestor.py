import yfinance as yf
import pandas as pd
import os
from typing import List, Dict, Any, Optional
from src.utils.config_loader import get_config_value, load_config

# Define the base path for Yahoo Finance raw data
YAHOO_FINANCE_RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../data/raw/yahoo_finance")

class YahooFinanceIngestor:
    """
    A class to ingest historical stock data from Yahoo Finance for different markets.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the YahooFinanceIngestor.

        Args:
            config (Optional[Dict[str, Any]]): A configuration dictionary. 
                                                If None, loads from the default config file.
        """
        if config is None:
            self.config = load_config() # Load default config if none provided
        else:
            self.config = config
        
        # Ensure the base Yahoo Finance directory exists
        if not os.path.exists(YAHOO_FINANCE_RAW_DATA_DIR):
            os.makedirs(YAHOO_FINANCE_RAW_DATA_DIR)
            print(f"Created base directory: {YAHOO_FINANCE_RAW_DATA_DIR}")

    def _get_market_config(self, market_key: str) -> Optional[Dict[str, Any]]:
        """
        Safely retrieves the configuration for a specific market (e.g., us_stocks, indian_stocks).
        """
        market_config = get_config_value(f"data_ingestion.yahoo_finance.{market_key}", config=self.config)
        if not market_config or not isinstance(market_config, dict):
            print(f"Warning: Configuration for market '{market_key}' is missing or invalid in config/config.yaml.")
            return None
        if not market_config.get("enabled", False):
            print(f"Info: Market '{market_key}' is disabled in the configuration.")
            return None
        return market_config

    def fetch_data_for_ticker(self, ticker: str, start_date: str, end_date: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetches historical data for a single stock ticker from Yahoo Finance.

        Args:
            ticker (str): The stock ticker symbol (e.g., "AAPL").
            start_date (str): The start date for the data (YYYY-MM-DD).
            end_date (str): The end date for the data (YYYY-MM-DD).
            interval (str): Data interval (e.g., "1d", "1h", "1wk"). Defaults to "1d".

        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame containing the historical data,
                                    or None if an error occurs.
        """
        try:
            print(f"Fetching data for {ticker} from {start_date} to {end_date} with interval {interval}...")
            stock = yf.Ticker(ticker)
            # Adding a timeout and retries might be good for robustness in a production system
            data = stock.history(start=start_date, end=end_date, interval=interval, auto_adjust=True)
            if data.empty:
                print(f"No data found for {ticker} for the given period and interval.")
                return None
            # Add ticker symbol as a column for easier identification if multiple dataframes are combined later
            data['Ticker'] = ticker 
            print(f"Successfully fetched {len(data)} rows for {ticker}.")
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    def save_data_to_csv(self, data: pd.DataFrame, ticker: str, market_subdir: str) -> bool:
        """
        Saves the DataFrame to a CSV file in the specified market subdirectory.

        Args:
            data (pd.DataFrame): The DataFrame to save.
            ticker (str): The stock ticker symbol, used for naming the file.
            market_subdir (str): The subdirectory for the market (e.g., "us_stocks").

        Returns:
            bool: True if data was saved successfully, False otherwise.
        """
        if data is None or data.empty:
            print(f"No data to save for {ticker}.")
            return False
        
        market_specific_path = os.path.join(YAHOO_FINANCE_RAW_DATA_DIR, market_subdir)
        if not os.path.exists(market_specific_path):
            os.makedirs(market_specific_path)
            print(f"Created directory: {market_specific_path}")
            
        file_path = os.path.join(market_specific_path, f"{ticker.replace('.NS', '_NS')}_raw_data.csv") # Sanitize .NS for filename
        try:
            data.to_csv(file_path)
            print(f"Data for {ticker} saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving data for {ticker} to CSV: {e}")
            return False

    def _ingest_market_data(self, market_key: str, market_config: Dict[str, Any]) -> None:
        """
        Ingests data for a specific market based on its configuration.
        """
        tickers: List[str] = market_config.get("tickers", [])
        start_date: str = market_config.get("start_date", "2020-01-01")
        end_date: str = market_config.get("end_date", "2023-12-31")
        interval: str = market_config.get("interval", "1d")

        if not tickers:
            print(f"No tickers found for market '{market_key}' in the configuration.")
            return

        print(f"--- Starting data ingestion for {market_key} --- ")
        for ticker in tickers:
            data = self.fetch_data_for_ticker(ticker, start_date, end_date, interval)
            if data is not None:
                self.save_data_to_csv(data, ticker, market_key) # market_key is used as subdir name
        print(f"--- {market_key} data ingestion completed. ---")

    def run_ingestion(self) -> None:
        """
        Runs the full data ingestion process for all configured and enabled markets.
        """
        print("Starting Yahoo Finance data ingestion for configured markets...")
        
        # Ingest US stocks
        us_market_config = self._get_market_config("us_stocks")
        if us_market_config:
            self._ingest_market_data("us_stocks", us_market_config)
            
        # Ingest Indian stocks
        indian_market_config = self._get_market_config("indian_stocks")
        if indian_market_config:
            self._ingest_market_data("indian_stocks", indian_market_config)
            
        # Add more markets here if needed by extending the config and calling _get_market_config & _ingest_market_data

        print("Yahoo Finance data ingestion process finished.")

# Example usage:
if __name__ == "__main__":
    ingestor = YahooFinanceIngestor()
    ingestor.run_ingestion() 