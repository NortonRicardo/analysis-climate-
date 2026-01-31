"""
Módulo para avaliar regressão linear usando os 20 vizinhos mais próximos.

Usa scikit-learn LinearRegression (Mínimos Quadrados Ordinários - OLS).

Usa os valores dos vizinhos (n1 a n20), suas distâncias e diferenças de altitude
como features para prever o valor observado.

Usa arquivos separados de treino (_train.parquet) e teste (_test.parquet).

Os resultados são salvos em CSV e podem ser acumulados para múltiplas variáveis.

Referência: Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.

pipenv run python train/linear_regression/linear_regression.py
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, List

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

def detect_variable_name(df: pd.DataFrame) -> str:
    """Detecta automaticamente o nome da variável no DataFrame."""
    exclude_patterns = ['code', 'time']
    
    for col in df.columns:
        if col in exclude_patterns:
            continue
        
        if col.startswith('n') and '_' in col:
            parts = col.split('_')
            if parts[0][1:].isdigit():
                continue
        
        if not (col.startswith('n') and col[1:].split('_')[0].isdigit()):
            return col
    
    raise ValueError("Não foi possível detectar a variável principal no DataFrame")


def prepare_features(
    df: pd.DataFrame, 
    var_name: str,
    n_neighbors: int = 20
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepara as features para regressão linear.
    
    Dados já estão normalizados entre 0 e 1, sem NaN.
    
    Features:
        - n1_{var} até n{n_neighbors}_{var}: valores dos vizinhos
        - n1_dist_km até n{n_neighbors}_dist_km: distâncias
        - n1_altdiff_km até n{n_neighbors}_altdiff_km: diferenças de altitude
    
    Returns:
        Tuple com (X, y, feature_names)
    """
    feature_cols = []
    
    for i in range(1, n_neighbors + 1):
        col_value = f'n{i}_{var_name}'
        col_dist = f'n{i}_dist_km'
        col_alt = f'n{i}_altdiff_km'
        
        if col_value in df.columns:
            feature_cols.append(col_value)
        if col_dist in df.columns:
            feature_cols.append(col_dist)
        if col_alt in df.columns:
            feature_cols.append(col_alt)
    
    target_col = var_name
    
    y = df[target_col].values
    X = df[feature_cols].values
    
    return X, y, feature_cols


def get_coefficients_df(model: LinearRegression, feature_names: List[str]) -> pd.DataFrame:
    """
    Extrai coeficientes do modelo sklearn.
    
    Args:
        model: Modelo LinearRegression treinado
        feature_names: Nomes das features
    
    Returns:
        DataFrame com coeficientes
    """
    return pd.DataFrame({
        'feature': ['intercept'] + list(feature_names),
        'coefficient': [model.intercept_] + list(model.coef_)
    })


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
    save_coefficients: bool = True
) -> pd.DataFrame:
    """
    Avalia regressão linear usando os vizinhos mais próximos (sklearn OLS).
    
    Usa arquivos separados de treino e teste:
        - {base_path}_train.parquet
        - {base_path}_test.parquet
    
    Args:
        base_path: Caminho base (ex: 'data_train/temperature/temperature')
        output_dir: Diretório de saída
        variable_name: Nome da variável (auto-detectado se None)
        n_neighbors: Número de vizinhos a usar (default: 20)
        save_coefficients: Se deve salvar os coeficientes
    
    Returns:
        DataFrame com os resultados das métricas
    """
    # Define paths dos arquivos
    train_path = f"{base_path}_train.parquet"
    test_path = f"{base_path}_test.parquet"
    
    # Lê arquivos de treino e teste
    print(f"Lendo arquivo de treino: {train_path}")
    df_train = pd.read_parquet(train_path)
    print(f"  → {len(df_train):,} registros de treino")
    
    print(f"Lendo arquivo de teste: {test_path}")
    df_test = pd.read_parquet(test_path)
    print(f"  → {len(df_test):,} registros de teste")
    
    # Detecta ou usa o nome da variável
    if variable_name is None:
        variable_name = detect_variable_name(df_train)
    print(f"  → Variável detectada: {variable_name}")
    
    # Prepara features de treino
    X_train, y_train, feature_names = prepare_features(df_train, variable_name, n_neighbors)
    print(f"  → {len(y_train):,} amostras de treino")
    print(f"  → {len(feature_names)} features: {n_neighbors} vizinhos × 3 (valor, dist, alt)")
    
    # Prepara features de teste
    X_test, y_test, _ = prepare_features(df_test, variable_name, n_neighbors)
    print(f"  → {len(y_test):,} amostras de teste")
    
    # Treina modelo (OLS - Mínimos Quadrados Ordinários)
    print("\nTreinando regressão linear (sklearn OLS)...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("  → Modelo treinado!")
    
    # Faz predições no conjunto de teste
    y_pred = model.predict(X_test)
    
    # Calcula métricas
    metrics = calculate_all_metrics(y_test, y_pred)
    metrics['variable'] = variable_name
    metrics['n_neighbors'] = n_neighbors
    metrics['n_features'] = len(feature_names)
    metrics['train_size'] = len(y_train)
    metrics['test_size'] = len(y_test)
    
    # Cria DataFrame de resultado
    result_df = pd.DataFrame([metrics])
    
    # Reordena colunas
    cols = ['variable', 'n_neighbors', 'n_features', 'train_size', 'test_size',
            'mae', 'rmse', 'bias', 'r', 'r2']
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
    
    # Salva coeficientes
    if save_coefficients:
        coef_df = get_coefficients_df(model, feature_names)
        coef_file = os.path.join(output_dir, f'coefficients_{variable_name}.csv')
        coef_df.to_csv(coef_file, index=False)
        print(f"✓ Coeficientes salvos em: {coef_file}")
    
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
    
    # Mostra top 10 coeficientes mais importantes (por valor absoluto)
    coef_df = get_coefficients_df(model, feature_names)
    coef_df['abs_coef'] = np.abs(coef_df['coefficient'])
    top_coefs = coef_df.nlargest(10, 'abs_coef')[['feature', 'coefficient']]
    
    print(f"\nTop 10 coeficientes mais importantes:")
    for _, row in top_coefs.iterrows():
        print(f"  {row['feature']:20s} {row['coefficient']:>10.4f}")
    
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
    evaluate_linear_regression(
        base_path='data_train/radiation/radiation',
        output_dir='train/linear_regression/results',
        n_neighbors=20
    )
