from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from data.preprocessing import DataPreprocessor



class CarPriceModel(BaseEstimator, TransformerMixin):
    def __init__(self):
        base_path = Path(__file__).parent.parent  # vai para o diretório src
        self.model_path = base_path / 'models' / 'car_price_catboost.pkl'
        self.preprocessor = DataPreprocessor()
        self.model = self._load_model()
    
    def _load_model(self):
        """Carrega o modelo XGBoost salvo"""
        try:
            return joblib.load(self.model_path)
        except Exception as e:
            raise Exception(f"Erro ao carregar o modelo: {str(e)}")
    
    
    def preprocess_data(self, data):
        """
        Pré-processa os dados de entrada usando o DataPreprocessor
        
        Args:
            data: Dict ou DataFrame com os dados de entrada
            
        Returns:
            DataFrame pré-processado
        """
        # Converte os dados de entrada em DataFrame se necessário
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Aplica o pré-processamento
        processed_data = self.preprocessor.preprocess_data(data, is_training=False)
        
        # Garante que todas as colunas necessárias estejam presentes
        required_columns = ['brand', 'model', 'milage', 'fuel_type', 'accident', 'engine_transmission', 'int_ext_color', 'car_age']
        
        missing_cols = set(required_columns) - set(processed_data.columns)
        if missing_cols:
            raise ValueError(f"Colunas ausentes nos dados: {missing_cols}")
            
        return processed_data[required_columns]
    
    def predict(self, X):
        """
        Realiza a predição usando o modelo XGBoost
        
        Args:
            X: Dict ou DataFrame com os dados de entrada
            
        Returns:
            Array com as predições
        """
        processed_data = self.preprocess_data(X)
        return self.model.predict(processed_data)
