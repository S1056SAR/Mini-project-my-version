import yaml
from dotenv import load_dotenv
import os
from typing import Dict, Any, Optional, List

# Load environment variables from .env file
load_dotenv()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")

def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """
    Loads the main configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
            return {}
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}")
        return {}

def get_config_value(key: str, default: Optional[Any] = None, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Retrieves a specific value from the loaded configuration using a dot-separated key.
    Example: get_config_value("data_ingestion.yahoo_finance.tickers")

    Args:
        key (str): The dot-separated key to retrieve (e.g., "parent.child.key").
        default (Optional[Any]): The default value to return if the key is not found.
        config (Optional[Dict[str, Any]]): The configuration dictionary. If None, loads the default config.

    Returns:
        Any: The configuration value or the default.
    """
    if config is None:
        config = load_config()

    keys = key.split('.')
    value = config
    try:
        for k in keys:
            if isinstance(value, dict):
                value = value[k]
            else: # If at any point value is not a dict and we still have keys, it's an error
                return default
        return value
    except KeyError:
        return default
    except TypeError: # Handles cases where a non-dict is indexed
        return default


def get_env_variable(var_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieves an environment variable.

    Args:
        var_name (str): The name of the environment variable.
        default (Optional[str]): The default value if the variable is not set.

    Returns:
        Optional[str]: The value of the environment variable or the default.
    """
    return os.getenv(var_name, default)

# Example usage (can be removed or commented out in production)
if __name__ == '__main__':
    config = load_config()
    print("Full config:", config)

    tickers = get_config_value("data_ingestion.yahoo_finance.tickers", config=config)
    print(f"Yahoo Finance Tickers: {tickers}")

    api_key = get_env_variable("KAGGLE_API_KEY", "not_set")
    print(f"Kaggle API Key: {api_key}")

    non_existent_value = get_config_value("some.non.existent.key", default="default_value", config=config)
    print(f"Non-existent config value: {non_existent_value}")

    specific_config_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
    specific_config = load_config(specific_config_path)
    print("Specifically loaded config version:", get_config_value("version", config=specific_config)) 