"""
Rede neural densa usando os vizinhos mais próximos.

Target: coluna n. Features: n1, n1_dist, n1_alt_diff, n1_idw, n2, ... (mesmo esquema da regressão linear).
Esquema do DF: measurement_time, code, n, n1, n1_dist, n1_alt_diff, n1_idw, n2, ...

Arquivos: {base_path}_train.parquet e _test.parquet (ex: data/data_train/temperature/).
Métricas: mae, rmse, bias, r, r2. Resultados em CSV; melhor modelo salvo por variável.

pipenv run python train/dense/dense.py
"""

import pandas as pd
import numpy as np
import os
import gc
from typing import Tuple, Optional, List

# TensorFlow imports
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


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
# FUNÇÕES AUXILIARES
# =============================================================================

def prepare_features(
    df: pd.DataFrame,
    n_neighbors: int = 20
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepara features para a rede densa: target = coluna n; por vizinho
    n1, n1_dist, n1_alt_diff, n1_idw, n2, ... (mesmo esquema da regressão linear).
    Esquema do DF: measurement_time, code, n, n1, n1_dist, n1_alt_diff, n1_idw, ...
    """
    if 'n' not in df.columns:
        raise ValueError("DataFrame deve ter coluna 'n' (target)")
    feature_cols = []
    for i in range(1, n_neighbors + 1):
        for suffix in ('', '_dist', '_alt_diff', '_idw'):
            col = f'n{i}{suffix}' if suffix else f'n{i}'
            if col in df.columns:
                feature_cols.append(col)
    X = df[feature_cols].values.astype(np.float32)
    y = df['n'].values.astype(np.float32)
    return X, y, feature_cols


# =============================================================================
# MODELO DE REDE NEURAL DENSA
# =============================================================================

def build_dense_model(
    input_dim: int,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01
) -> Sequential:
    """
    Constrói modelo de rede neural densa.
    
    Arquitetura fixa:
        Input -> 128 -> 128 -> 64 -> 32 -> 16 -> 1
    
    Usa AdamW com weight decay (regularização L2 desacoplada).
    
    Args:
        input_dim: Número de features de entrada
        learning_rate: Taxa de aprendizado
        weight_decay: Regularização L2 via AdamW (default: 0.01)
    
    Returns:
        Modelo Keras compilado
    """
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    # Compila modelo com AdamW
    model.compile(
        optimizer=AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def evaluate_dense(
    base_path: str,
    output_dir: str = 'train/dense/results',
    models_dir: str = 'train/dense/models',
    variable_name: Optional[str] = None,
    n_neighbors: int = 20,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    epochs: int = 100,
    batch_size: int = 512,
    validation_split: float = 0.15,
    patience: int = 15
) -> pd.DataFrame:
    """
    Avalia rede neural densa usando os vizinhos mais próximos.
    
    Usa features estruturadas (ponderadas por distância e altitude).
    Usa AdamW com weight decay para regularização.
    
    Usa arquivos separados de treino e teste:
        - {base_path}_train.parquet
        - {base_path}_test.parquet
    
    Args:
        base_path: Caminho base (ex: 'data/data_train/temperature/temperature')
        output_dir: Diretório de saída para métricas
        models_dir: Diretório para salvar modelos
        variable_name: Nome da variável para o CSV (extraído do base_path se None)
        n_neighbors: Número de vizinhos a usar (default: 20)
        learning_rate: Taxa de aprendizado (default: 0.001)
        weight_decay: Regularização L2 via AdamW (default: 0.01)
        epochs: Número máximo de épocas
        batch_size: Tamanho do batch (default: 512)
        validation_split: Fração do treino para validação (default: 0.15)
        patience: Épocas sem melhora antes de parar (default: 15)
    
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
    
    # Libera memória do DataFrame de treino (já extraímos X_train e y_train)
    del df_train
    gc.collect()
    
    # Cria diretórios
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Caminho do modelo
    model_path = os.path.join(models_dir, f'dense_{variable_name}.keras')
    
    # Constrói modelo
    print(f"\nConstruindo rede neural densa...")
    print(f"  → Arquitetura: 128 → 128 → 64 → 32 → 16 → 1")
    model = build_dense_model(len(feature_names), learning_rate, weight_decay)
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Treina modelo
    print(f"\nTreinando modelo (AdamW)...")
    print(f"  → Épocas máximas: {epochs}")
    print(f"  → Batch size: {batch_size}")
    print(f"  → Learning rate: {learning_rate}")
    print(f"  → Weight decay: {weight_decay}")
    print(f"  → Validação: {validation_split*100:.0f}% do treino")
    print(f"  → Early stopping: {patience} épocas sem melhora")
    print()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    # Carrega melhor modelo
    print(f"\nCarregando melhor modelo de: {model_path}")
    model = load_model(model_path)
    
    # Libera memória dos dados de treino (não precisamos mais)
    del X_train, y_train
    gc.collect()
    
    # Lê arquivo de teste (apenas quando necessário)
    print(f"\nLendo arquivo de teste: {test_path}")
    df_test = pd.read_parquet(test_path)
    print(f"  → {len(df_test):,} registros de teste")
    
    X_test, y_test, _ = prepare_features(df_test, n_neighbors)
    del df_test  # Libera memória
    gc.collect()
    
    # Faz predições no teste
    print("Fazendo predições no conjunto de teste...")
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=0).flatten()
    
    metrics = calculate_all_metrics(y_test, y_pred)
    metrics['variable'] = variable_name
    result_df = pd.DataFrame([metrics])
    cols = ['variable', 'mae', 'rmse', 'bias', 'r', 'r2']
    result_df = result_df[cols]
    
    # Salva métricas
    results_file = os.path.join(output_dir, 'dense_metrics.csv')
    
    if os.path.exists(results_file):
        existing_df = pd.read_csv(results_file)
        existing_df = existing_df[existing_df['variable'] != variable_name]
        result_df = pd.concat([existing_df, result_df], ignore_index=True)
    
    result_df.to_csv(results_file, index=False)
    
    # Mostra resultados
    print(f"\n{'='*60}")
    print(f"REDE NEURAL DENSA PARA: {variable_name}")
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

# base_path='data/data_train/temperature/temperature'
# base_path='data/data_train/humidity/humidity'
# base_path='data/data_train/radiation/radiation'
# base_path='data/data_train/pressure/pressure'
# base_path='data/data_train/rainfall/rainfall'

if __name__ == '__main__':
    evaluate_dense(
        base_path='data/data_train/temperature/temperature',
        output_dir='train/dense/results',
        models_dir='train/dense/models',
        learning_rate=0.001,
        weight_decay=0.01,
        epochs=100,
        batch_size=512,
        validation_split=0.15,
        patience=15
    )
