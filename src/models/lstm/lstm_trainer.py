import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, Optional, Tuple
import joblib # Added for saving the scaler

from .lstm_data_handler import LSTMDataHandler
from .lstm_model import LSTMModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODELS_SAVE_DIR = os.path.join(PROJECT_ROOT, "experiments", "lstm_models")

class LSTMTrainer:
    """
    Handles the training and evaluation of the LSTM model for a specific stock.
    """
    def __init__(self, config: Dict, ticker: str, market_type: str):
        """
        Initializes the LSTMTrainer.

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
        logging.info(f"Using device: {self.device}")

        # Set random seed for reproducibility
        seed = config.get('random_seed', 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        self.data_handler = LSTMDataHandler(config, ticker, market_type)
        self.model: Optional[LSTMModel] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None # Though not used in this trainer's core loop yet
        self.target_scaler = None

        self._model_save_path = os.path.join(MODELS_SAVE_DIR, self.market_type, self.ticker)
        os.makedirs(self._model_save_path, exist_ok=True)

    def _initialize_components(self) -> bool:
        """Initializes data loaders, model, optimizer, and criterion."""
        logging.info(f"Initializing components for {self.ticker}...")
        loaders_output = self.data_handler.prepare_data_loaders()
        if not loaders_output:
            logging.error(f"Failed to prepare data loaders for {self.ticker}. Cannot initialize trainer.")
            return False
        
        self.train_loader, self.val_loader, self.test_loader, self.target_scaler = loaders_output

        if not self.train_loader or not self.val_loader or len(self.train_loader) == 0 or len(self.val_loader) == 0:
            logging.error(f"Train or Validation DataLoader is empty for {self.ticker}. Training cannot proceed.")
            return False

        # Determine input_features from the data
        try:
            X_sample_batch, _ = next(iter(self.train_loader))
            input_features = X_sample_batch.shape[2]
            logging.info(f"Determined input_features: {input_features} for {self.ticker}")
        except StopIteration:
            logging.error(f"Train loader for {self.ticker} is empty, cannot determine input features.")
            return False
        except Exception as e:
            logging.error(f"Error getting sample batch from train_loader for {self.ticker}: {e}")
            return False

        # Model
        model_params_from_config = self.lstm_global_config.copy() # Get LSTM specific hyperparams
        self.model = LSTMModel(input_features=input_features, model_config=model_params_from_config).to(self.device)
        logging.info(f"LSTMModel initialized for {self.ticker}.")

        # Optimizer
        lr = self.lstm_global_config.get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        logging.info(f"Optimizer Adam initialized with lr: {lr} for {self.ticker}.")

        # Criterion (Loss Function)
        self.criterion = nn.MSELoss()
        logging.info(f"Criterion MSELoss initialized for {self.ticker}.")
        return True

    def _train_epoch(self) -> float:
        """Runs a single training epoch."""
        self.model.train() # Set model to training mode
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Optional: Log batch loss
            # if batch_idx % 10 == 0: # Log every 10 batches
            #     logging.debug(f'Epoch {self.current_epoch_num} Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')

        return epoch_loss / len(self.train_loader)

    def _evaluate_epoch(self) -> float:
        """Runs a single evaluation epoch on the validation set."""
        self.model.eval() # Set model to evaluation mode
        epoch_loss = 0.0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                epoch_loss += loss.item()
        return epoch_loss / len(self.val_loader)

    def train(self) -> Optional[str]:
        """
        Orchestrates the training process for the configured number of epochs.
        Returns path to the best model if training was successful, else None.
        """
        if not self._initialize_components(): # Initialize all parts before training
            return None

        epochs = self.lstm_global_config.get('epochs', 100)
        best_val_loss = float('inf')
        best_model_path = None
        best_scaler_path = None # Added for scaler path
        patience = self.lstm_global_config.get('early_stopping_patience', 10)
        patience_counter = 0

        logging.info(f"Starting training for {self.ticker} for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            self.current_epoch_num = epoch # For potential batch logging
            train_loss = self._train_epoch()
            val_loss = self._evaluate_epoch()

            logging.info(f"Ticker: {self.ticker} | Epoch: {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_filename = "best_model.pth"
                scaler_filename = "target_scaler.pkl" # Added for scaler
                best_model_path = os.path.join(self._model_save_path, model_filename)
                best_scaler_path = os.path.join(self._model_save_path, scaler_filename) # Added for scaler

                torch.save(self.model.state_dict(), best_model_path)
                if self.target_scaler: # Ensure scaler exists
                    joblib.dump(self.target_scaler, best_scaler_path) # Save the scaler
                    logging.info(f"Validation loss improved for {self.ticker}. Saved best model to {best_model_path} and scaler to {best_scaler_path}")
                else:
                    logging.warning(f"Validation loss improved for {self.ticker}, model saved to {best_model_path}, but target_scaler was not found/saved.")
                patience_counter = 0 # Reset patience
            else:
                patience_counter +=1
            
            # Save latest model (optional)
            # latest_model_path = os.path.join(self._model_save_path, "latest_model.pth")
            # torch.save(self.model.state_dict(), latest_model_path)

            if patience_counter >= patience:
                logging.info(f"Early stopping triggered for {self.ticker} at epoch {epoch} due to no improvement in validation loss for {patience} epochs.")
                break
        
        if best_model_path: # We now care if the model path was set, implying scaler might have been too
            logging.info(f"Training finished for {self.ticker}. Best validation loss: {best_val_loss:.6f}")
            # Return model path, the caller can infer scaler path if needed
            return best_model_path 
        else:
            logging.warning(f"Training completed for {self.ticker}, but no best model was saved (possibly due to no improvement or issues).")
            return None

if __name__ == '__main__':
    print("Testing LSTMTrainer...")
    # This requires a config file and actual featured data to exist for the test ticker.
    # Ensure previous pipelines (ingestion, preprocessing, feature engineering) have run.
    
    PROJECT_ROOT_TEST = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    import sys
    sys.path.insert(0, PROJECT_ROOT_TEST) 
    from src.utils.config_loader import load_config # type: ignore

    test_config = load_config()
    if not test_config:
        print("Failed to load config for LSTMTrainer test. Ensure config/config.yaml is present.")
    else:
        example_ticker = 'AAPL' # Ensure this ticker has data from previous steps
        example_market = 'us_stocks'
        
        # Quick check if featured data exists for the test ticker
        featured_file_path = os.path.join(PROJECT_ROOT_TEST, "data", "featured", "yahoo_finance", example_market, f"{example_ticker}_featured_data.csv")
        if not os.path.exists(featured_file_path):
            print(f"Featured data for {example_ticker} not found at {featured_file_path}. Run previous pipelines first. Skipping LSTMTrainer test.")
        else:
            # Modify config for a quick test run if desired
            # test_config['models']['lstm']['epochs'] = 3 
            # test_config['models']['lstm']['sequence_length'] = 30 
            # test_config['models']['lstm']['batch_size'] = 16
            print(f"Attempting to train LSTM model for {example_ticker}...")
            trainer = LSTMTrainer(config=test_config, ticker=example_ticker, market_type=example_market)
            best_model_loc = trainer.train()
            if best_model_loc:
                print(f"LSTMTrainer test completed for {example_ticker}. Best model saved at: {best_model_loc}")
            else:
                print(f"LSTMTrainer test for {example_ticker} completed, but training might have failed or no model was saved.")

    print("LSTMTrainer test finished.") 