# Main entry point for the FinTech AI Trading application
import argparse
import os # Added for path manipulation if needed later
import sys 
# Calculate the project root directory (one level up from 'src')
# __file__ is the path to the current script (src/main.py)
# os.path.dirname(__file__) is the directory of the current script (src/)
# os.path.join(..., '..') goes one level up to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to Python's module search path if it's not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Added to potentially manipulate Python path if necessary

# Ensure the src directory is in the Python path to allow for absolute imports
# This is often necessary when running scripts from the root directory or via certain IDEs.
# current_dir = os.path.dirname(os.path.abspath(__file__))
# src_path = os.path.dirname(current_dir) # This assumes main.py is in src/
# if src_path not in sys.path:
#     sys.path.insert(0, src_path)

from src.utils.config_loader import load_config, get_config_value
from src.data_ingestion.yahoo_finance_ingestor import YahooFinanceIngestor
from src.preprocessing.yahoo_finance_preprocessor import YahooFinancePreprocessor
from src.feature_engineering.yahoo_finance_feature_engineer import YahooFinanceFeatureEngineer
from src.models.lstm.lstm_trainer import LSTMTrainer
from src.models.lstm.lstm_predictor import LSTMPredictor
# Import other modules as they are developed
# e.g., from src.preprocessing.main_preprocessor import MainPreprocessor
# e.g., from src.models.lstm.trainer import LSTMTrainer

def main():
    """
    Main function to orchestrate the different pipelines of the application.
    Parses command-line arguments to determine which pipeline to run.
    """
    parser = argparse.ArgumentParser(description="FinTech AI Trading Application")
    parser.add_argument(
        "--pipeline",
        choices=["data_ingestion_yahoo", "preprocess_yahoo", "feature_engineering_yahoo", "train_lstm", "predict_lstm", "preprocess", "train_rl", "evaluate", "trade"],
        required=True,
        help="Specify the pipeline to run."
    )
    # Add other arguments as needed, e.g., --config-file
    # config_file_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml") # Default path from src
    # parser.add_argument("--config-file", default=config_file_path, help="Path to the main configuration file.")
    parser.add_argument("--ticker", type=str, help="Specify a single ticker to run the pipeline for (e.g., AAPL for train_lstm or predict_lstm). Optional for some, required for predict_lstm.")
    parser.add_argument("--market", type=str, choices=["us_stocks", "indian_stocks"], help="Specify the market for the ticker (us_stocks or indian_stocks). Required if --ticker is used for train_lstm or predict_lstm.")

    args = parser.parse_args()

    # Load the main configuration
    # config_loader.py uses a path relative to its own location to find config.yaml
    # If you run main.py from the project root, this relative path should work.
    config = load_config()
    if not config:
        print("Failed to load configuration. Ensure config/config.yaml exists and is valid. Exiting.")
        return

    print(f"Running pipeline: {args.pipeline}")
    project_version = get_config_value('version', config=config, default="N/A")
    print(f"Project Version from config: {project_version}")

    if args.pipeline == "data_ingestion_yahoo":
        print("Initializing Yahoo Finance data ingestion pipeline...")
        # Pass the loaded config to the ingestor
        yahoo_ingestor = YahooFinanceIngestor(config=config) 
        yahoo_ingestor.run_ingestion()
    elif args.pipeline == "preprocess_yahoo":
        print("Initializing Yahoo Finance data preprocessing pipeline...")
        yahoo_preprocessor = YahooFinancePreprocessor(config=config)
        yahoo_preprocessor.run_preprocessing()
    elif args.pipeline == "feature_engineering_yahoo":
        print("Initializing Yahoo Finance feature engineering pipeline...")
        yahoo_feature_engineer = YahooFinanceFeatureEngineer(config=config)
        yahoo_feature_engineer.run_feature_engineering()
    elif args.pipeline == "train_lstm":
        print("Initializing LSTM model training pipeline...")
        yahoo_config = config.get('data_ingestion', {}).get('yahoo_finance', {})
        markets_to_process = [] # Changed variable name for clarity

        if args.ticker and args.market:
            print(f"Training LSTM for a single ticker: {args.ticker} in market: {args.market}")
            if yahoo_config.get(args.market, {}).get('enabled', False) and args.ticker in yahoo_config.get(args.market, {}).get('tickers', []):
                markets_to_process.append((args.market, [args.ticker]))
            else:
                print(f"Error: Ticker {args.ticker} or market {args.market} not found or not enabled in config for training.")
                return
        elif args.ticker and not args.market:
            print("Error: --market (us_stocks or indian_stocks) is required when --ticker is specified for train_lstm.")
            return
        else:
            print("Training LSTM for all configured and enabled tickers...")
            if yahoo_config.get('us_stocks', {}).get('enabled', False):
                markets_to_process.append(('us_stocks', yahoo_config['us_stocks'].get('tickers', [])))
            if yahoo_config.get('indian_stocks', {}).get('enabled', False):
                markets_to_process.append(('indian_stocks', yahoo_config['indian_stocks'].get('tickers', [])))

        if not markets_to_process:
            print("No markets or tickers are enabled or specified for LSTM training in the configuration.")
            return

        for market_name, tickers_list in markets_to_process:
            print(f"--- Training LSTM models for market: {market_name} ---")
            if not tickers_list:
                print(f"No tickers configured for market: {market_name}. Skipping.")
                continue
            for ticker_symbol in tickers_list:
                print(f"== Starting LSTM training for ticker: {ticker_symbol} in {market_name} ==")
                trainer = LSTMTrainer(config=config, ticker=ticker_symbol, market_type=market_name)
                best_model_path = trainer.train()
                if best_model_path:
                    print(f"Successfully trained LSTM for {ticker_symbol}. Best model saved at: {best_model_path}")
                else:
                    print(f"LSTM training failed or no model was saved for {ticker_symbol}.")
                print(f"== Finished LSTM training for ticker: {ticker_symbol} in {market_name} ==\n")
        print("LSTM model training pipeline finished.")
    elif args.pipeline == "predict_lstm":
        print("Initializing LSTM model prediction pipeline...")
        if not args.ticker or not args.market:
            print("Error: --ticker and --market are required for the predict_lstm pipeline.")
            return

        # Validate ticker and market from config (optional but good practice)
        yahoo_config = config.get('data_ingestion', {}).get('yahoo_finance', {})
        market_config = yahoo_config.get(args.market, {})
        if not market_config.get('enabled', False) or args.ticker not in market_config.get('tickers', []):
            # We might still want to predict for a ticker even if it's not in the main list,
            # as long as its model and data exist. This check might be too restrictive.
            # For now, we proceed if model/data exist, this is more for user guidance.
            print(f"Warning: Ticker {args.ticker} or market {args.market} may not be in the primary config list, or market not enabled. Proceeding if model/data exists.")

        print(f"== Generating predictions for ticker: {args.ticker} in market {args.market} ==")
        predictor = LSTMPredictor(config=config, ticker=args.ticker, market_type=args.market)
        predictions_df = predictor.predict()
        if predictions_df is not None:
            print(f"Successfully generated predictions for {args.ticker}. Results (head):")
            print(predictions_df.head())
            # The predictor already saves the file, message is in the predictor logs.
        else:
            print(f"Prediction failed for {args.ticker}.")
        print(f"== Finished LSTM prediction for ticker: {args.ticker} in {args.market} ==\n")
        print("LSTM model prediction pipeline finished.")
    elif args.pipeline == "preprocess":
        print("General preprocessing pipeline not yet implemented. Use 'preprocess_yahoo' for Yahoo Finance data.")
        # preprocessor = MainPreprocessor(config)
        # preprocessor.run()
    elif args.pipeline == "train_rl":
        print("Reinforcement Learning training pipeline not yet implemented.")
        # rl_trainer = RLTrainer(config)
        # rl_trainer.train()
    elif args.pipeline == "evaluate":
        print("Evaluation pipeline not yet implemented.")
        # evaluator = Evaluator(config)
        # evaluator.evaluate()
    elif args.pipeline == "trade":
        print("Trade execution pipeline not yet implemented.")
        # trader = Trader(config)
        # trader.trade()
    else:
        print(f"Pipeline '{args.pipeline}' is not recognized or not yet implemented.")

if __name__ == "__main__":
    main() 