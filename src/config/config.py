import os
from pathlib import Path

# Diretórios base
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Subdiretórios de dados
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Configurações do modelo
MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "n_estimators": 100
}

# Configurações da aplicação
APP_CONFIG = {
    "model_path": os.path.join(MODELS_DIR, "car_price_model.pkl"),
    "scaler_path": os.path.join(MODELS_DIR, "scaler.pkl")
} 