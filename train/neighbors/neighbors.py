"""
Módulo para avaliar a qualidade da imputação por vizinhos mais próximos.

Calcula métricas comparando o valor observado com o valor do vizinho mais próximo (n1).

Usa arquivo de teste (_test.parquet) para avaliação consistente com outros modelos.

Os resultados são salvos em CSV e podem ser acumulados para múltiplas variáveis.

pipenv run python train/neighbors/neighbors.py
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional


# =============================================================================
# MÉTRICAS (implementação manual, sem dependências externas)
# =============================================================================

def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error (Erro Médio Absoluto)
    
    MAE = (1/n) * Σ|y_true - y_pred|
    
    Interpretação: Erro médio real em unidades da variável.
    """
    return np.mean(np.abs(y_true - y_pred))


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error (Raiz do Erro Quadrático Médio)
    
    RMSE = sqrt((1/n) * Σ(y_true - y_pred)²)
    
    Interpretação: Penaliza erros extremos mais do que o MAE.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calc_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Bias (Viés/Tendência)
    
    Bias = (1/n) * Σ(y_pred - y_true)
    
    Interpretação:
    - Positivo: modelo tende a superestimar
    - Negativo: modelo tende a subestimar
    - Zero: sem viés sistemático
    """
    return np.mean(y_pred - y_true)


def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coeficiente de Determinação (R²)
    
    R² = 1 - (SS_res / SS_tot)
    onde:
        SS_res = Σ(y_true - y_pred)²  (soma dos quadrados dos resíduos)
        SS_tot = Σ(y_true - mean(y_true))²  (soma total dos quadrados)
    
    Interpretação:
    - 1.0: predição perfeita
    - 0.0: predição igual à média
    - <0: predição pior que a média
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return np.nan
    
    return 1 - (ss_res / ss_tot)


def calc_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coeficiente de Correlação de Pearson (r)
    
    r = Σ((x - mean_x)(y - mean_y)) / sqrt(Σ(x - mean_x)² * Σ(y - mean_y)²)
    
    Interpretação:
    - 1.0: correlação positiva perfeita
    - 0.0: sem correlação linear
    - -1.0: correlação negativa perfeita
    """
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    numerator = np.sum((y_true - mean_true) * (y_pred - mean_pred))
    denominator = np.sqrt(
        np.sum((y_true - mean_true) ** 2) * np.sum((y_pred - mean_pred) ** 2)
    )
    
    if denominator == 0:
        return np.nan
    
    return numerator / denominator


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def detect_variable_name(df: pd.DataFrame) -> str:
    """
    Detecta automaticamente o nome da variável no DataFrame.
    
    Procura por colunas que não sejam: code, time, ou que comecem com 'n' seguido de número.
    """
    exclude_patterns = ['code', 'time']
    
    for col in df.columns:
        # Ignora colunas de código e tempo
        if col in exclude_patterns:
            continue
        
        # Ignora colunas de vizinhos (n1_*, n2_*, etc.)
        if col.startswith('n') and '_' in col:
            parts = col.split('_')
            if parts[0][1:].isdigit():
                continue
        
        # Se não é vizinho e não é code/time, é a variável principal
        if not (col.startswith('n') and col[1:].split('_')[0].isdigit()):
            return col
    
    raise ValueError("Não foi possível detectar a variável principal no DataFrame")


def get_valid_pairs(
    df: pd.DataFrame, 
    var_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Obtém pares válidos de valores observados e preditos (n1).
    Remove registros onde qualquer um dos valores é NaN.
    
    Returns:
        Tuple com (y_true, y_pred) como arrays numpy
    """
    observed_col = var_name
    neighbor_col = f'n1_{var_name}'
    
    if neighbor_col not in df.columns:
        raise ValueError(f"Coluna do vizinho '{neighbor_col}' não encontrada no DataFrame")
    
    # Cria DataFrame com as duas colunas
    pairs = df[[observed_col, neighbor_col]].copy()
    
    # Remove NaN
    pairs_valid = pairs.dropna()
    
    y_true = pairs_valid[observed_col].values
    y_pred = pairs_valid[neighbor_col].values
    
    return y_true, y_pred


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula todas as métricas de avaliação.
    
    Returns:
        Dicionário com todas as métricas
    """
    return {
        'n_samples': len(y_true),
        'mae': calc_mae(y_true, y_pred),
        'rmse': calc_rmse(y_true, y_pred),
        'bias': calc_bias(y_true, y_pred),
        'r': calc_correlation(y_true, y_pred),
        'r2': calc_r2(y_true, y_pred),
        'mean_observed': np.mean(y_true),
        'mean_predicted': np.mean(y_pred),
        'std_observed': np.std(y_true),
        'std_predicted': np.std(y_pred),
    }


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def evaluate_neighbors(
    base_path: str,
    output_dir: str = 'train/neighbors/results',
    variable_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Avalia a qualidade da imputação por vizinho mais próximo.
    
    Usa o arquivo de teste para avaliação:
        - {base_path}_test.parquet
    
    Args:
        base_path: Caminho base (ex: 'data_train/temperature/temperature')
                   Vai carregar {base_path}_test.parquet
        output_dir: Diretório onde salvar os resultados (CSV)
        variable_name: Nome da variável (auto-detectado se None)
    
    Returns:
        DataFrame com os resultados das métricas
    """
    # Define path do arquivo de teste
    test_path = f"{base_path}_test.parquet"
    
    # Lê o arquivo de teste
    print(f"Lendo arquivo de teste: {test_path}")
    df = pd.read_parquet(test_path)
    print(f"  → {len(df):,} registros carregados")
    
    # Detecta ou usa o nome da variável
    if variable_name is None:
        variable_name = detect_variable_name(df)
    print(f"  → Variável detectada: {variable_name}")
    
    # Obtém pares válidos
    y_true, y_pred = get_valid_pairs(df, variable_name)
    print(f"  → {len(y_true):,} pares válidos para avaliação")
    
    # Calcula métricas
    metrics = calculate_all_metrics(y_true, y_pred)
    metrics['variable'] = variable_name
    
    # Cria DataFrame de resultado
    result_df = pd.DataFrame([metrics])
    
    # Reordena colunas (variável primeiro)
    cols = ['variable', 'n_samples', 'mae', 'rmse', 'bias', 'r', 'r2', 
            'mean_observed', 'mean_predicted', 'std_observed', 'std_predicted']
    result_df = result_df[cols]
    
    # Cria diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Caminho do arquivo de resultados
    results_file = os.path.join(output_dir, 'neighbors_metrics.csv')
    
    # Se arquivo já existe, lê e adiciona/atualiza
    if os.path.exists(results_file):
        existing_df = pd.read_csv(results_file)
        
        # Remove entrada anterior da mesma variável (se existir)
        existing_df = existing_df[existing_df['variable'] != variable_name]
        
        # Concatena com novo resultado
        result_df = pd.concat([existing_df, result_df], ignore_index=True)
    
    # Salva arquivo CSV
    result_df.to_csv(results_file, index=False)
    print(f"\n✓ Resultados salvos em: {results_file}")
    
    # Mostra resultados
    print(f"\n{'='*60}")
    print(f"MÉTRICAS PARA: {variable_name}")
    print(f"{'='*60}")
    print(f"  Amostras válidas:  {metrics['n_samples']:,}")
    print(f"  MAE:               {metrics['mae']:.4f}")
    print(f"  RMSE:              {metrics['rmse']:.4f}")
    print(f"  Bias:              {metrics['bias']:.4f}")
    print(f"  r (correlação):    {metrics['r']:.4f}")
    print(f"  R²:                {metrics['r2']:.4f}")
    print(f"{'='*60}")
    
    return result_df


# =============================================================================
# EXECUÇÃO
# =============================================================================

# base_path='data_train/temperature/temperature'
# base_path='data_train/humidity/humidity'
# base_path='data_train/radiation/radiation'

# base_path='data_train/pressure/pressure'
# base_path='data_train/rainfall/rainfall'

if __name__ == '__main__':
    evaluate_neighbors(
        base_path='data_train/radiation/radiation',
        output_dir='train/neighbors/results'
    )
