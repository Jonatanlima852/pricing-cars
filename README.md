# 🚗 Previsão de Preços de Carros

Sistema de previsão de preços de carros usando Machine Learning com interface web em Streamlit.

## 📋 Pré-requisitos

- Python 3.8+
- pip (gerenciador de pacotes Python)
- Git
- Conhecimento básico de linha de comando

## 🚀 Configuração do Ambiente

### Clonando o Repositório

1. Clone o repositório 
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd previsao-precos-carros
```

### Configurando o Ambiente Virtual

#### Windows

```bash
python -m venv .venv
```	

#### Ative o ambiente virtual

```bash
.venv\Scripts\activate
```
#### Instalando as Dependências

```bash
pip install -r requirements.txt
```
#### Linux / MacOS

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

#### Instalando as Dependências

```bash
pip install -r requirements.txt
```

### Configurando Variáveis de Ambiente (opcional)
Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:

PYTHONPATH=src/
DEBUG=True

## 💻 Desenvolvimento

### Estrutura do Código

O projeto segue uma estrutura modular:

previsao-precos-carros/
│
├── data/ # Diretório para dados
│ ├── raw/ # Dados brutos
│ └── processed/ # Dados processados
│
├── notebooks/ # Jupyter notebooks para análises
│ └── exploratory/ # Análises exploratórias
│
├── src/ # Código fonte
│ ├── app/ # Aplicação Streamlit
│ ├── data/ # Scripts de processamento de dados
│ ├── models/ # Scripts de modelos ML
│ ├── config/ # Configurações
│ └── utils/ # Funções utilitárias
│
├── tests/ # Testes unitários
├── requirements.txt # Dependências do projeto
└── README.md # Este arquivo



### Executando a Aplicação

1. Ative o ambiente virtual (conforme instruções acima)

2. Execute a aplicação Streamlit:

```bash
streamlit run src/app/main.py
```


3. Acesse a aplicação em `http://localhost:8501`

### Executando Testes

```bash
pytest
```	

#### Executando Testes com Coverage

```bash
pytest --cov=src tests/
```


## 📊 Dados

### Preparação dos Dados

1. Coloque seus dados brutos em `data/raw/`
2. Execute o processamento:

```bash
python src/data/preprocessing.py
```

### Treinamento do Modelo

1. Execute o treinamento:

```bash
python src/models/train_model.py
```


## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature

```bash
git checkout -b feature/minha-feature
```

3. Faça suas alterações e adicione testes

4. Faça o push para sua branch

```bash
git push origin feature/minha-feature
```

5. Crie um pull request
