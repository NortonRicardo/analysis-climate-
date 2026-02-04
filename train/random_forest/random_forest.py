"""
Random Forest (sklearn) usando os vizinhos mais próximos.

Target: coluna n. Features: n1, n1_dist, n1_alt_diff, n1_idw, n2, ... (por vizinho).
Esquema do DF: measurement_time, code, n, n1, n1_dist, n1_alt_diff, n1_idw, n2, ...

Mesmos dados de train/test que o script denso: base_path em data_dense ou data_train.
Carregamento por batch (parquet iter_batches). Métricas: mae, rmse, bias, r, r2.
Resultados em CSV; modelo salvo com joblib.

pipenv run python train/random_forest/random_forest.py
"""

import pandas as pd
import numpy as np
import os
import gc
from typing import Tuple, Optional, List
import pyarrow.parquet as pq
import joblib
from sklearn.ensemble import RandomForestRegressor

# Usar mais CPU: mesmo número de núcleos que o script denso (7 para treino, 1 para sistema)
_N_JOBS = 7  # 8 vCPUs na máquina; 7 para treino, 1 para sistema

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

# =============================================================================
# MÉTRICAS (implementação manual, mesmas do script denso)
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


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula todas as métricas de avaliação (mesmas do script denso)."""
    return {
        'mae': calc_mae(y_true, y_pred),
        'rmse': calc_rmse(y_true, y_pred),
        'bias': calc_bias(y_true, y_pred),
        'r': calc_correlation(y_true, y_pred),
        'r2': calc_r2(y_true, y_pred),
    }


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def prepare_features(
    df: pd.DataFrame,
    n_neighbors: int = 20
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepara features: target = coluna n; por vizinho
    n1, n1_dist, n1_alt_diff, n1_idw, n2, ... (mesmo esquema do script denso).
    """
    if 'n' not in df.columns:
        raise ValueError("DataFrame deve ter coluna 'n' (target)")
    feature_cols = []
    for i in range(1, n_neighbors + 1):
        for suffix in ('', '_dist', '_alt_diff', '_idw'):
            col = f'n{i}{suffix}' if suffix else f'n{i}'
            if col in df.columns:
                feature_cols.append(col)
    X = df[feature_cols].values.astype(np.float64)
    y = df['n'].values.astype(np.float64)
    return X, y, feature_cols


def infer_feature_columns_from_parquet(
    path: str,
    n_neighbors: int = 20
) -> Tuple[List[str], int]:
    """
    Infere as colunas de features e o número total de linhas do parquet
    sem carregar todo o arquivo em memória.
    """
    parquet_file = pq.ParquetFile(path)
    columns = parquet_file.schema.names

    feature_cols: List[str] = []
    for i in range(1, n_neighbors + 1):
        for suffix in ('', '_dist', '_alt_diff', '_idw'):
            col = f'n{i}{suffix}' if suffix else f'n{i}'
            if col in columns:
                feature_cols.append(col)

    n_rows = parquet_file.metadata.num_rows
    return feature_cols, n_rows


def load_parquet_in_batches(
    path: str,
    feature_cols: List[str],
    read_batch_size: int = 100_000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega parquet em batches e retorna X, y concatenados.
    Mesma lógica de carregamento por batch do script denso (iter_batches).
    """
    list_X: List[np.ndarray] = []
    list_y: List[np.ndarray] = []

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=read_batch_size):
        df_batch = batch.to_pandas()
        if 'n' not in df_batch.columns:
            raise ValueError("DataFrame deve ter coluna 'n' (target)")
        X_batch = df_batch[feature_cols].values.astype(np.float64)
        y_batch = df_batch['n'].values.astype(np.float64)
        list_X.append(X_batch)
        list_y.append(y_batch)
        del df_batch, batch
        gc.collect()

    del pf
    gc.collect()

    X = np.vstack(list_X)
    y = np.concatenate(list_y)
    del list_X, list_y
    gc.collect()

    return X, y


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def evaluate_random_forest(
    base_path: str,
    output_dir: str = 'train/random_forest/results',
    models_dir: str = 'train/random_forest/models',
    variable_name: Optional[str] = None,
    n_neighbors: int = 20,
    n_estimators: int = 100,
    max_depth: Optional[int] = 20,
    min_samples_leaf: int = 5,
    read_batch_size: int = 100_000
) -> pd.DataFrame:
    """
    Avalia Random Forest usando os vizinhos mais próximos (mesmos dados do script denso).

    base_path em data_dense (ex: data/data_dense/pressure/pressure) ou data_train.
    Treino e teste: {base_path}_train.parquet, _test.parquet.
    Carregamento por batch (iter_batches). Métricas: mae, rmse, bias, r, r2.

    Args:
        base_path: Caminho base (ex: 'data/data_dense/pressure/pressure')
        output_dir: Diretório de saída para métricas
        models_dir: Diretório para salvar modelo (joblib)
        variable_name: Nome da variável para o CSV (extraído do base_path se None)
        n_neighbors: Número de vizinhos (default: 20)
        n_estimators: Número de árvores (default: 100)
        max_depth: Profundidade máxima das árvores (default: 20)
        min_samples_leaf: Mínimo de amostras por folha (default: 5)
        read_batch_size: Linhas lidas por batch do parquet (default: 100_000)

    Returns:
        DataFrame com os resultados das métricas
    """
    train_path = f"{base_path}_train.parquet"
    test_path = f"{base_path}_test.parquet"

    # Fallback para data_train se não existir em data_dense (mesmo que o script denso)
    if not os.path.isfile(train_path):
        train_path_fallback = base_path.replace('data_dense', 'data_train') + '_train.parquet'
        if os.path.isfile(train_path_fallback):
            train_path = train_path_fallback
    if not os.path.isfile(test_path):
        test_path_fallback = base_path.replace('data_dense', 'data_train') + '_test.parquet'
        if os.path.isfile(test_path_fallback):
            test_path = test_path_fallback

    if variable_name is None:
        variable_name = os.path.basename(base_path.rstrip('/'))

    print(f"Variável: {variable_name}")

    if not os.path.isfile(train_path):
        raise FileNotFoundError(
            f"Arquivo de treino não encontrado: {train_path} "
            "(nem em data_dense nem em data_train)"
        )
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Arquivo de teste não encontrado: {test_path}")

    # Infere features sem carregar o arquivo inteiro
    feature_names, n_train_rows = infer_feature_columns_from_parquet(
        train_path, n_neighbors=n_neighbors
    )
    print(f"Treino: {train_path}")
    print(f"  → {n_train_rows:,} registros (carregamento por batch)")
    print(f"  → {len(feature_names)} features")

    # Carrega treino em batches
    X_train, y_train = load_parquet_in_batches(
        train_path, feature_names, read_batch_size=read_batch_size
    )
    print(f"  → X_train {X_train.shape}, y_train {y_train.shape}")

    # Treina Random Forest (n_jobs = 7, equivalente ao _n_cpus do script denso)
    print(f"\nTreinando Random Forest (n_estimators={n_estimators}, n_jobs={_N_JOBS})...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=_N_JOBS,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("  → Modelo treinado!")

    del X_train, y_train
    gc.collect()

    # Carrega teste em batches
    print(f"\nTeste: {test_path}")
    X_test, y_test = load_parquet_in_batches(
        test_path, feature_names, read_batch_size=read_batch_size
    )
    print(f"  → {len(y_test):,} registros de teste")

    # Predições
    print("Fazendo predições no conjunto de teste...")
    y_pred = model.predict(X_test)

    metrics = calculate_all_metrics(y_test, y_pred)
    metrics['variable'] = variable_name
    result_df = pd.DataFrame([metrics])
    cols = ['variable', 'mae', 'rmse', 'bias', 'r', 'r2']
    result_df = result_df[cols]

    # Diretórios e salvamento
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    results_file = os.path.join(output_dir, 'random_forest_metrics.csv')
    if os.path.exists(results_file):
        existing_df = pd.read_csv(results_file)
        existing_df = existing_df[existing_df['variable'] != variable_name]
        result_df = pd.concat([existing_df, result_df], ignore_index=True)
    result_df.to_csv(results_file, index=False)

    model_path = os.path.join(models_dir, f'random_forest_{variable_name}.joblib')
    joblib.dump(model, model_path)

    # Resultados
    print(f"\n{'='*60}")
    print(f"RANDOM FOREST PARA: {variable_name}")
    print(f"{'='*60}")
    print(f"  MAE:               {metrics['mae']:.4f}")
    print(f"  RMSE:              {metrics['rmse']:.4f}")
    print(f"  Bias:              {metrics['bias']:.4f}")
    print(f"  r (correlação):    {metrics['r']:.4f}")
    print(f"  R²:                {metrics['r2']:.4f}")
    print(f"{'='*60}")
    print(f"\n✓ Métricas salvas em: {results_file}")
    print(f"✓ Modelo salvo em: {model_path}")

    return result_df


# =============================================================================
# EXECUÇÃO
# =============================================================================

# base_path='data/data_dense/temperature/temperature'
# base_path='data/data_dense/humidity/humidity'
# base_path='data/data_dense/radiation/radiation'
# base_path='data/data_dense/pressure/pressure'
# base_path='data/data_dense/rainfall/rainfall'

if __name__ == '__main__':
    evaluate_random_forest(
        base_path='../../data/data_dense/pressure/pressure',
        output_dir='train/random_forest/results',
        models_dir='train/random_forest/models',
        n_neighbors=20,
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=5,
        read_batch_size=100_000
    )
