from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from data.preprocessing import DataPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor


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
    
    def save_model(self):
        """Salva o modelo treinado"""
        try:
            joblib.dump(self.model, self.model_path)
        except Exception as e:
            raise Exception(f"Erro ao salvar o modelo: {str(e)}")
    
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
    
    def fit(self, X, y):
        """
        Treina o modelo (se necessário)
        
        Args:
            X: Features de treino
            y: Target
        """
        processed_X = self.preprocess_data(X)
        self.model.fit(processed_X, y)
        return self
    
    def train(self, train_data: pd.DataFrame):
        """
        Treina o modelo CatBoost usando os dados preprocessados
        
        Args:
            train_data: DataFrame com os dados de treino, incluindo a coluna 'price'
        """
        # Separar features e target
        X = train_data.drop(columns=['price', 'id'], errors='ignore')
        y = train_data['price']
        
        # Preprocessar dados
        X_processed = self.preprocess_data(X)
        
        # Split train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # Inicializar modelo CatBoost
        self.model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=10,
            random_state=42,
            early_stopping_rounds=50,
            verbose=100
        )
        
        # Treinar modelo
        self.model.fit(
            X_train, 
            y_train,
            eval_set=(X_test, y_test),
            use_best_model=True
        )
        
        # Avaliar modelo
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"CatBoost RMSE: {rmse:.2f}")
        print(f"CatBoost R²: {r2:.2f}")
        
        # Salvar modelo
        self.save_model()
        
        return self