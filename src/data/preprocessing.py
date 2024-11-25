import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
import category_encoders as ce  # Importar TargetEncoder

class DataPreprocessor:
    """
    Classe para pré-processamento dos dados de carros
    """
    def __init__(self):
        base_path = Path(__file__).parent.parent  # vai para o diretório src
        encoder_path = base_path / 'models' / 'target_encoder.pkl'  # Atualizado para TargetEncoder.pkl
        scaler_path = base_path / 'models' / 'standard_scaler.pkl'
        # Atualizando a ordem para corresponder ao CSV original
        self.feature_order = [
            'brand',
            'model',
            'car_age',
            'milage',
            'fuel_type',
            'engine_transmission',
            'int_ext_color',
            'accident',
            'clean_title'
        ]
        
        # Carrega o encoder salvo ou cria um novo se não existir
        try:
            self.target_encoder = joblib.load(encoder_path)
        except:
            print("Não foi possível carregar o encoder salvo. Inicializando um novo.")
            self.target_encoder = ce.TargetEncoder(cols=[])  # Inicializar sem colunas, será definido no método
        

        # Inicializar o scaler
        try:
            self.scaler = joblib.load(scaler_path)
        except:
            print("Não foi possível carregar o scaler salvo. Inicializando um novo.")
            self.scaler = StandardScaler()
        
    def knn_impute(self, df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
        """
        Realiza imputação de dados usando KNN
        
        Args:
            df: DataFrame com os dados
            n_neighbors: Número de vizinhos para KNN
            
        Returns:
            DataFrame com dados imputados
        """
        df_encoded = df.copy()
        
        # Codifica variáveis categóricas para KNN
        for col in df_encoded.select_dtypes(include='object').columns:
            df_encoded[col] = df_encoded[col].astype('category').cat.codes
            
        # Aplica KNN Imputer
        knn_imputer = KNNImputer(n_neighbors=n_neighbors)
        df_imputed = pd.DataFrame(
            knn_imputer.fit_transform(df_encoded), 
            columns=df_encoded.columns
        )
        
        # Reverte codificação para variáveis categóricas
        for col in df.select_dtypes(include='object').columns:
            df_imputed[col] = df_imputed[col].round().astype(int).map(
                dict(enumerate(df[col].astype('category').cat.categories))
            )
            
        return df_imputed

    def remove_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Remove outliers usando método IQR
        
        Args:
            df: DataFrame com os dados
            column: Nome da coluna para remover outliers
            
        Returns:
            DataFrame sem outliers
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria novas features
        
        Args:
            df: DataFrame com os dados
            
        Returns:
            DataFrame com novas features
        """
        df_new = df.copy()
    
        # Cria features compostas
        df_new['engine_transmission'] = df_new['engine'].astype(str) + "_" + df_new['transmission'].astype(str)
        df_new['int_ext_color'] = df_new['int_col'].astype(str) + "_" + df_new['ext_col'].astype(str)
    
        # Remove colunas originais usadas nas features compostas
        df_new.drop(columns=['engine', 'transmission', 'int_col', 'ext_col'], inplace=True, errors='ignore')

        # Remover a coluna 'clean_title' se existir
        if 'clean_title' in df_new.columns:
            df_new.drop(columns=['clean_title'], inplace=True)
        
        return df_new

    
    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Realiza todo o pré-processamento dos dados
        """
        df_processed = df.copy()
        
        
        # Imputação de dados
        df_processed = self.knn_impute(df_processed, n_neighbors=25)

        # Remove ID se existir
        if 'id' in df_processed.columns:
            df_processed = df_processed.drop(columns=['id'])
            
        # Calcular car_age se necessário
        if 'model_year' in df_processed.columns:
            df_processed['car_age'] = 2024 - df_processed['model_year'].astype(int)
            df_processed = df_processed.drop(columns=['model_year'], errors='ignore')
        
        
        # Criar features compostas
        df_processed = self.create_features(df_processed)

        
        # Identificar colunas categóricas após criação das features
        cat_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        

        # Carregar o encoder salvo
        encoder_path = Path(__file__).parent.parent / 'models' / 'target_encoder.pkl'
        self.target_encoder = joblib.load(encoder_path)
        df_processed[cat_cols] = self.target_encoder.transform(df_processed[cat_cols])
        
        # Escalonamento das features numéricas
        
        numeric_cols = ['brand', 'model', 'car_age', 'milage', 'fuel_type',
                       'engine_transmission', 'int_ext_color', 'accident']
        

        # Carregar o scaler salvo
        scaler_path = Path(__file__).parent.parent / 'models' / 'standard_scaler.pkl'
        self.scaler = joblib.load(scaler_path)
        df_processed[numeric_cols] = self.scaler.transform(df_processed[numeric_cols])
        
        # Selecionar apenas as colunas necessárias
        required_columns = ['brand', 'model', 'car_age', 'milage', 'fuel_type',
                            'engine_transmission', 'int_ext_color', 'accident']
        df_processed = df_processed[required_columns]
        
        # Garantir que todas as colunas estão como float
        for col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(float)
        
        # Debug: Valores finais

        

        return df_processed
        
        
    
    def prepare_for_training(self, df: pd.DataFrame, target_col: str = 'price', 
                           scale_features: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara dados para treinamento
        """
        X, y = self.preprocess_data(df, is_training=True)
        return X, y
