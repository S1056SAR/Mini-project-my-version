import torch
import torch.nn as nn
from typing import Dict

class LSTMModel(nn.Module):
    """
    Defines the LSTM model architecture for stock price forecasting.
    """
    def __init__(self, input_features: int, model_config: Dict):
        """
        Initializes the LSTMModel.

        Args:
            input_features (int): The number of input features for each time step
                                  (e.g., Open, High, Low, Close, Volume, MA_10, RSI, etc.).
            model_config (Dict): A dictionary containing model-specific hyperparameters,
                                 expected to be from the main config file under 'models.lstm'.
                                 Expected keys:
                                 - hidden_units (int): Number of units in LSTM hidden layers.
                                 - hidden_layers (int): Number of LSTM layers (maps to nn.LSTM num_layers).
                                 - dropout_rate (float): Dropout rate for LSTM layers (if num_layers > 1).
                                 - output_size (int): Number of output values (typically 1 for next price).
        """
        super(LSTMModel, self).__init__()
        self.hidden_units = model_config.get('hidden_units', 50)
        self.num_lstm_layers = model_config.get('hidden_layers', 2)
        self.dropout_rate = model_config.get('dropout_rate', 0.2)
        self.output_size = model_config.get('output_size', 1)

        # LSTM layer
        # Batch_first=True means input/output tensors are (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=self.hidden_units,
            num_layers=self.num_lstm_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_lstm_layers > 1 else 0
        )

        # Fully connected layer to predict the output value
        self.fc = nn.Linear(self.hidden_units, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Initialize hidden state and cell state with zeros
        # h0 shape: (num_layers * num_directions, batch_size, hidden_size)
        # c0 shape: (num_layers * num_directions, batch_size, hidden_size)
        # Ensure device compatibility if using GPU later
        device = x.device
        h0 = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_units).to(device)
        c0 = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_units).to(device)

        # Pass input through LSTM layer
        # lstm_out shape: (batch_size, sequence_length, hidden_units)
        # self.lstm returns: output, (hn, cn)
        lstm_out, _ = self.lstm(x, (h0, c0)) # We don't need the hidden states here for basic prediction

        # We only need the output from the last time step of the sequence for prediction
        # lstm_out shape: (batch_size, sequence_length, hidden_units)
        # last_time_step_out shape: (batch_size, hidden_units)
        last_time_step_out = lstm_out[:, -1, :] 

        # Pass the output of the last time step through the fully connected layer
        out = self.fc(last_time_step_out)
        return out

if __name__ == '__main__':
    # Example Usage and Test
    print("Testing LSTMModel...")
    
    # Dummy model configuration (mirroring what might be in config.yaml)
    dummy_model_config = {
        'hidden_units': 64, 
        'hidden_layers': 2,  # Changed from num_layers to hidden_layers to match config
        'dropout_rate': 0.2, 
        'output_size': 1      
    }
    input_features_count = 10 # Example: OHLCV + 5 technical indicators
    
    model = LSTMModel(input_features=input_features_count, model_config=dummy_model_config)
    print("Model architecture:")
    print(model)

    # Create a dummy input tensor
    batch_size = 5
    sequence_length = 60
    dummy_input = torch.randn(batch_size, sequence_length, input_features_count) # (batch, seq_len, features)
    
    # Perform a forward pass
    try:
        output = model(dummy_input)
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}") # Expected: (batch_size, output_size)
        assert output.shape == (batch_size, dummy_model_config['output_size'])
        print("LSTMModel test successful.")
    except Exception as e:
        print(f"Error during LSTMModel test: {e}") 