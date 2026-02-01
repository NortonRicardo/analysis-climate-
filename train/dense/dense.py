"""
Módulo para avaliar rede neural densa usando os 20 vizinhos mais próximos.

Usa os valores dos vizinhos (n1 a n20), suas distâncias e diferenças de altitude
como features para prever o valor observado.

Usa arquivos separados de treino (_train.parquet) e teste (_test.parquet).

Os resultados são salvos em CSV e o melhor modelo é salvo para cada variável.

pipenv run python train/dense/dense.py
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, List

# TensorFlow imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model


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
    Prepara as features para a rede neural.
    
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


# =============================================================================
# MODELO DE REDE NEURAL DENSA
# =============================================================================

def build_dense_model(
    input_dim: int,
    architecture: str = 'medium'
) -> Sequential:
    """
    Constrói modelo de rede neural densa.
    
    Arquiteturas disponíveis:
        - 'small': 64 -> 32 -> 1 (rápido, menos parâmetros)
        - 'medium': 128 -> 64 -> 32 -> 1 (balanceado)
        - 'large': 256 -> 128 -> 64 -> 32 -> 1 (mais capacidade)
    
    Args:
        input_dim: Número de features de entrada
        architecture: Tipo de arquitetura ('small', 'medium', 'large')
    
    Returns:
        Modelo Keras compilado
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    if architecture == 'small':
        # ~2.5k parâmetros para 60 features
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        
    elif architecture == 'medium':
        # ~12k parâmetros para 60 features
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        
    elif architecture == 'large':
        # ~50k parâmetros para 60 features
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
    
    else:
        raise ValueError(f"Arquitetura desconhecida: {architecture}")
    
    # Compila modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),
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
    architecture: str = 'medium',
    epochs: int = 100,
    batch_size: int = 4096,
    validation_split: float = 0.1,
    patience: int = 10
) -> pd.DataFrame:
    """
    Avalia rede neural densa usando os vizinhos mais próximos.
    
    Usa arquivos separados de treino e teste:
        - {base_path}_train.parquet
        - {base_path}_test.parquet
    
    Args:
        base_path: Caminho base (ex: 'data_train/temperature/temperature')
        output_dir: Diretório de saída para métricas
        models_dir: Diretório para salvar modelos
        variable_name: Nome da variável (auto-detectado se None)
        n_neighbors: Número de vizinhos a usar (default: 20)
        architecture: Arquitetura da rede ('small', 'medium', 'large')
        epochs: Número máximo de épocas
        batch_size: Tamanho do batch (maior = mais rápido, mas mais memória)
        validation_split: Fração do treino para validação
        patience: Épocas sem melhora antes de parar (EarlyStopping)
    
    Returns:
        DataFrame com os resultados das métricas
    """
    # Define paths dos arquivos
    train_path = f"{base_path}_train.parquet"
    test_path = f"{base_path}_test.parquet"
    
    # Lê arquivos
    print(f"Lendo arquivo de treino: {train_path}")
    df_train = pd.read_parquet(train_path)
    print(f"  → {len(df_train):,} registros de treino")
    
    print(f"Lendo arquivo de teste: {test_path}")
    df_test = pd.read_parquet(test_path)
    print(f"  → {len(df_test):,} registros de teste")
    
    # Detecta variável
    if variable_name is None:
        variable_name = detect_variable_name(df_train)
    print(f"  → Variável detectada: {variable_name}")
    
    # Prepara features
    X_train, y_train, feature_names = prepare_features(df_train, variable_name, n_neighbors)
    print(f"  → {len(y_train):,} amostras de treino")
    print(f"  → {len(feature_names)} features")
    
    X_test, y_test, _ = prepare_features(df_test, variable_name, n_neighbors)
    print(f"  → {len(y_test):,} amostras de teste")
    
    # Cria diretórios
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Caminho do modelo
    model_path = os.path.join(models_dir, f'dense_{variable_name}.keras')
    
    # Constrói modelo
    print(f"\nConstruindo rede neural densa ({architecture})...")
    model = build_dense_model(len(feature_names), architecture)
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
    print(f"\nTreinando modelo...")
    print(f"  → Épocas máximas: {epochs}")
    print(f"  → Batch size: {batch_size:,}")
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
    
    # Faz predições no teste
    print("Fazendo predições no conjunto de teste...")
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=0).flatten()
    
    # Calcula métricas
    metrics = calculate_all_metrics(y_test, y_pred)
    metrics['variable'] = variable_name
    metrics['n_neighbors'] = n_neighbors
    metrics['architecture'] = architecture
    metrics['train_size'] = len(y_train)
    metrics['test_size'] = len(y_test)
    metrics['epochs_trained'] = len(history.history['loss'])
    metrics['best_val_loss'] = min(history.history['val_loss'])
    
    # Cria DataFrame
    result_df = pd.DataFrame([metrics])
    
    # Reordena colunas
    cols = ['variable', 'architecture', 'n_neighbors', 'train_size', 'test_size',
            'epochs_trained', 'mae', 'rmse', 'bias', 'r', 'r2']
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

# base_path='data_train/temperature/temperature'
# base_path='data_train/humidity/humidity'
# base_path='data_train/radiation/radiation'

# base_path='data_train/pressure/pressure'
# base_path='data_train/rainfall/rainfall'

if __name__ == '__main__':
    evaluate_dense(
        base_path='data_train/temperature/temperature',
        output_dir='train/dense/results',
        models_dir='train/dense/models',
        architecture='large',  # 'small', 'medium', 'large'
        epochs=100,
        batch_size=4096,
        patience=10
    )
