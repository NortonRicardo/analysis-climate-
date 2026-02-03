"""
Módulo para avaliar rede neural densa usando os 20 vizinhos mais próximos.

Cria features com estrutura:
    - Pondera valores dos vizinhos pelo inverso da distância (vizinhos próximos pesam mais)
    - Cria interações entre valor, distância e altitude

Usa arquivos separados de treino (_train.parquet) e teste (_test.parquet).

Os resultados são salvos em CSV e o melhor modelo é salvo para cada variável.

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
    Prepara as features para a rede neural COM ESTRUTURA.
    
    Usa Inverse Distance Weighting (IDW) para ponderar valores dos vizinhos.
    
    Para cada vizinho i, cria features:
        - valor do vizinho (n{i}_{var})
        - IDW: valor / (dist + 1e-6)  → vizinhos próximos pesam mais
        - distância em km
        - diferença de altitude em km
    
    Features agregadas:
        - IDW total (soma ponderada / soma dos pesos)
        - Média simples dos vizinhos
        - Variância dos vizinhos
    
    Returns:
        Tuple com (X, y, feature_names)
    """
    features = []
    feature_names = []
    
    # Arrays para cálculos agregados
    valores = []
    pesos_idw = []
    
    for i in range(1, n_neighbors + 1):
        col_value = f'n{i}_{var_name}'
        col_dist = f'n{i}_dist_km'
        col_alt = f'n{i}_altdiff_km'
        
        if col_value not in df.columns:
            continue
            
        val = df[col_value].values
        dist = df[col_dist].values if col_dist in df.columns else np.ones(len(df))
        alt = df[col_alt].values if col_alt in df.columns else np.zeros(len(df))
        
        # Feature 1: Valor bruto do vizinho
        features.append(val)
        feature_names.append(f'n{i}_value')
        
        # Feature 2: IDW - valor ponderado pelo inverso da distância
        # Quanto menor a distância, maior o peso
        peso_idw = 1.0 / (dist + 1e-6)
        val_idw = val / (dist + 1e-6)
        features.append(val_idw)
        feature_names.append(f'n{i}_idw')
        
        # Feature 3: Distância em km
        features.append(dist)
        feature_names.append(f'n{i}_dist')
        
        # Feature 4: Diferença de altitude em km
        features.append(alt)
        feature_names.append(f'n{i}_alt')
        
        # Guarda para agregados
        valores.append(val)
        pesos_idw.append(peso_idw)
    
    # Features agregadas
    valores = np.array(valores)  # (n_neighbors, n_samples)
    pesos_idw = np.array(pesos_idw)
    
    # IDW total: Σ(valor * peso) / Σ(peso)
    soma_pesos = np.sum(pesos_idw, axis=0) + 1e-8
    idw_total = np.sum(valores * pesos_idw, axis=0) / soma_pesos
    features.append(idw_total)
    feature_names.append('idw_total')
    
    # Média simples dos vizinhos
    media_simples = np.mean(valores, axis=0)
    features.append(media_simples)
    feature_names.append('simple_mean')
    
    # Variância dos vizinhos (indica dispersão espacial)
    variancia = np.var(valores, axis=0)
    features.append(variancia)
    feature_names.append('variance')
    
    # Diferença entre n1 e média (o quanto o vizinho mais próximo difere)
    diff_n1_media = valores[0] - media_simples
    features.append(diff_n1_media)
    feature_names.append('n1_diff_mean')
    
    # Stack features (usa float32 para economizar memória)
    X = np.column_stack(features).astype(np.float32)
    y = df[var_name].values.astype(np.float32)
    
    return X, y, feature_names


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
        BatchNormalization(),
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
        base_path: Caminho base (ex: 'data_train/temperature/temperature')
        output_dir: Diretório de saída para métricas
        models_dir: Diretório para salvar modelos
        variable_name: Nome da variável (auto-detectado se None)
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
    # Define paths dos arquivos
    train_path = f"{base_path}_train.parquet"
    test_path = f"{base_path}_test.parquet"
    
    # Lê arquivo de treino
    print(f"Lendo arquivo de treino: {train_path}")
    df_train = pd.read_parquet(train_path)
    print(f"  → {len(df_train):,} registros de treino")
    
    # Detecta variável
    if variable_name is None:
        variable_name = detect_variable_name(df_train)
    print(f"  → Variável detectada: {variable_name}")
    
    # Prepara features com estrutura IDW
    print("\nPreparando features com IDW (Inverse Distance Weighting)...")
    X_train, y_train, feature_names = prepare_features(df_train, variable_name, n_neighbors)
    train_size = len(y_train)  # Guarda antes de deletar
    print(f"  → {train_size:,} amostras de treino")
    print(f"  → {len(feature_names)} features (valor, IDW, dist, alt por vizinho + agregadas)")
    
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
    
    # Prepara features de teste
    X_test, y_test, _ = prepare_features(df_test, variable_name, n_neighbors)
    test_size = len(y_test)  # Guarda antes de deletar df_test
    del df_test  # Libera memória
    gc.collect()
    
    # Faz predições no teste
    print("Fazendo predições no conjunto de teste...")
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=0).flatten()
    
    # Calcula métricas
    metrics = calculate_all_metrics(y_test, y_pred)
    metrics['variable'] = variable_name
    metrics['n_neighbors'] = n_neighbors
    metrics['n_features'] = len(feature_names)
    metrics['learning_rate'] = learning_rate
    metrics['train_size'] = train_size
    metrics['test_size'] = test_size
    metrics['epochs_trained'] = len(history.history['loss'])
    metrics['best_val_loss'] = min(history.history['val_loss'])
    
    # Cria DataFrame
    result_df = pd.DataFrame([metrics])
    
    # Reordena colunas
    cols = ['variable', 'n_neighbors', 'n_features', 'train_size', 'test_size',
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
        learning_rate=0.001,
        weight_decay=0.01,
        epochs=100,
        batch_size=512,
        validation_split=0.15,
        patience=15
    )
