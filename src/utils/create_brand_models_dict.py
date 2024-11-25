import pandas as pd
import json
from pathlib import Path

def create_brand_models_dict():
    """
    Cria e salva o dicionário de marcas e modelos
    """
    # Ajuste o caminho conforme necessário
    base_path = Path(__file__).parent.parent
    csv_path = base_path.parent / 'notebooks' / 'data' / 'raw' / 'train.csv'
    output_path = base_path / 'utils' / 'brand_models.json'
    
    # Criar diretório se não existir
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ler CSV e criar dicionário
    df = pd.read_csv(csv_path)
    brand_models = {}
    
    for brand in df['brand'].unique():
        models = df[df['brand'] == brand]['model'].unique().tolist()
        brand_models[brand] = sorted(models)
    
    # Adicionar "Other" se necessário
    if "Other" not in brand_models:
        brand_models["Other"] = ["Other"]
    
    # Salvar como JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(brand_models, f, ensure_ascii=False, indent=4)
    
    print(f"Dicionário salvo em: {output_path}")

if __name__ == "__main__":
    create_brand_models_dict()
