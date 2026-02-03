"""
Regressão linear (sklearn OLS) usando os vizinhos mais próximos.

Target: coluna n. Features: n1, n1_dist, n1_alt_diff, n1_idw, n2, ... (por vizinho).
Esquema do DF: measurement_time, code, n, n1, n1_dist, n1_alt_diff, n1_idw, n2, ...

Arquivos: {base_path}_train.parquet e _test.parquet (ex: data/data_train/temperature/).
Métricas: mae, rmse, bias, r, r2. Resultados em CSV.

pipenv run python train/linear_regression/linear_regression.py
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, List

import joblib
from sklearn.linear_model import LinearRegression


# =============================================================================
# MÉTRICAS (implementação manual, sem dependências externas)
# =============================================================================

def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calc_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Bias (Viés/Tendência)"""
    return np.mean(y_pred - y_true)


def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coeficiente de Determinação (R²)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return np.nan
    
    return 1 - (ss_res / ss_tot)


def calc_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coeficiente de Correlação de Pearson (r)"""
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

def prepare_features(
    df: pd.DataFrame,
    n_neighbors: int = 20
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepara features para regressão: target = coluna n; features = n1, n1_dist,
    n1_alt_diff, n1_idw, n2, ... por vizinho (esquema: n, n1, n1_dist, n1_alt_diff, n1_idw, ...).
    """
    if 'n' not in df.columns:
        raise ValueError("DataFrame deve ter coluna 'n' (target)")
    feature_cols = []
    for i in range(1, n_neighbors + 1):
        for suffix in ('', '_dist', '_alt_diff', '_idw'):
            col = f'n{i}{suffix}' if suffix else f'n{i}'
            if col in df.columns:
                feature_cols.append(col)
    y = df['n'].values
    X = df[feature_cols].values
    return X, y, feature_cols


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula todas as métricas de avaliação."""
    return {
        'mae': calc_mae(y_true, y_pred),
        'rmse': calc_rmse(y_true, y_pred),
        'bias': calc_bias(y_true, y_pred),
        'r': calc_correlation(y_true, y_pred),
        'r2': calc_r2(y_true, y_pred),
    }


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def evaluate_linear_regression(
    base_path: str,
    output_dir: str = 'train/linear_regression/results',
    variable_name: Optional[str] = None,
    n_neighbors: int = 20,
    save_model: bool = True
) -> pd.DataFrame:
    """
    Avalia regressão linear usando os vizinhos mais próximos (sklearn OLS).
    
    Usa arquivos separados de treino e teste:
        - {base_path}_train.parquet
        - {base_path}_test.parquet
    
    Args:
        base_path: Caminho base (ex: 'data/data_train/temperature/temperature')
        output_dir: Diretório de saída
        variable_name: Nome da variável para o CSV (extraído do base_path se None)
        n_neighbors: Número de vizinhos a usar (default: 20)
        save_model: Se deve salvar o modelo treinado (joblib)
    
    Returns:
        DataFrame com os resultados das métricas
    """
    train_path = f"{base_path}_train.parquet"
    test_path = f"{base_path}_test.parquet"
    
    if variable_name is None:
        variable_name = os.path.basename(base_path.rstrip('/'))
    print(f"Variável: {variable_name}")
    
    print(f"Lendo treino: {train_path}")
    df_train = pd.read_parquet(train_path)
    print(f"  → {len(df_train):,} registros")
    
    X_train, y_train, feature_names = prepare_features(df_train, n_neighbors)
    print(f"  → {len(y_train):,} amostras, {len(feature_names)} features")
    
    print("\nTreinando regressão linear (sklearn OLS)...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("  → Modelo treinado!")
    
    del df_train, X_train, y_train  # libera memória antes de carregar teste
    
    print(f"Lendo teste: {test_path}")
    df_test = pd.read_parquet(test_path)
    X_test, y_test, _ = prepare_features(df_test, n_neighbors)
    print(f"  → {len(y_test):,} amostras de teste")
    
    y_pred = model.predict(X_test)
    
    metrics = calculate_all_metrics(y_test, y_pred)
    metrics['variable'] = variable_name
    
    result_df = pd.DataFrame([metrics])
    cols = ['variable', 'mae', 'rmse', 'bias', 'r', 'r2']
    result_df = result_df[cols]
    
    # Cria diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva métricas
    results_file = os.path.join(output_dir, 'linear_regression_metrics.csv')
    
    if os.path.exists(results_file):
        existing_df = pd.read_csv(results_file)
        existing_df = existing_df[existing_df['variable'] != variable_name]
        result_df = pd.concat([existing_df, result_df], ignore_index=True)
    
    result_df.to_csv(results_file, index=False)
    print(f"\n✓ Métricas salvas em: {results_file}")
    
    if save_model:
        model_file = os.path.join(output_dir, f'model_{variable_name}.joblib')
        joblib.dump(model, model_file)
        print(f"✓ Modelo salvo em: {model_file}")
    
    # Mostra resultados
    print(f"\n{'='*60}")
    print(f"REGRESSÃO LINEAR (OLS) PARA: {variable_name}")
    print(f"{'='*60}")
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

# base_path='data/data_train/temperature/temperature'
# base_path='data/data_train/humidity/humidity'
# base_path='data/data_train/radiation/radiation'
# base_path='data/data_train/pressure/pressure'
# base_path='data/data_train/rainfall/rainfall'

if __name__ == '__main__':
    evaluate_linear_regression(
        base_path='data/data_train/temperature/temperature',
        output_dir='train/linear_regression/results',
        n_neighbors=20
    )
