"""
Rede neural densa usando os vizinhos mais próximos.

Target: coluna n. Features: n1, n1_dist, n1_alt_diff, n1_idw, n2, ... (mesmo esquema da regressão linear).
Esquema do DF: measurement_time, code, n, n1, n1_dist, n1_alt_diff, n1_idw, n2, ...

base_path em data_dense (ex: data/data_dense/pressure/pressure). Arquivos: _train, _val, _test.parquet.
Se _val não existir, é gerado a partir de data/data_train/{...}/_train.parquet (batches); depois começa o treino com monitor='val_loss'.
Métricas: mae, rmse, bias, r, r2. Resultados em CSV; melhor modelo salvo por variável.

pipenv run python train/dense/dense.py
"""

# Generator já fornece dados na ordem do parquet (sem shuffle)

import pandas as pd
import numpy as np
import os
import gc
from typing import Tuple, Optional, List
import pyarrow as pa
import pyarrow.parquet as pq

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Usar mais CPU: mais threads = mais uso de núcleos (menos máquina ociosa)
_n_cpus = 7  # 8 vCPUs na máquina; 7 para treino, 1 para sistema
tf.config.threading.set_intra_op_parallelism_threads(_n_cpus)
tf.config.threading.set_inter_op_parallelism_threads(_n_cpus)

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()  # notebook: usa diretório atual

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


def ensure_val_parquet(
    val_path: str,
    base_path: str,
    validation_fraction: float = 0.15,
    read_batch_size: int = 50_000
) -> None:
    """
    Se val_path não existir, gera *_val.parquet a partir de
    data/data_train/{...}/{...}_train.parquet (substitui data_dense por data_train
    no base_path), lendo em batches e escrevendo ~validation_fraction das linhas.
    Libera memória ao final.
    """
    if os.path.isfile(val_path):
        return
    # base_path está em data_dense (ex: data/data_dense/pressure/pressure)
    # origem do train: data/data_train/pressure/pressure_train.parquet
    source_base = base_path.replace('data_dense', 'data_train')
    source_train_path = f"{source_base}_train.parquet"
    if not os.path.isfile(source_train_path):
        raise FileNotFoundError(
            f"Val não encontrado em {val_path} e não foi possível gerar: "
            f"origem {source_train_path} não existe."
        )
    print(f"  → Val não encontrado em data_dense. Gerando a partir de: {source_train_path}")
    os.makedirs(os.path.dirname(val_path) or '.', exist_ok=True)
    pf = pq.ParquetFile(source_train_path)
    schema = pf.schema_arrow
    writer = pq.ParquetWriter(val_path, schema)
    global_idx = 0
    for batch in pf.iter_batches(batch_size=read_batch_size):
        n = batch.num_rows
        # 15% das linhas para val (determinístico: índice % 100 < 15)
        val_indices = [
            i for i in range(n)
            if (global_idx + i) % 100 < int(validation_fraction * 100)
        ]
        if val_indices:
            val_batch = batch.take(pa.array(val_indices))
            writer.write_batch(val_batch)
        global_idx += n
        del batch, val_indices
        gc.collect()
    writer.close()
    del pf, writer
    gc.collect()
    print(f"  → Val salvo em: {val_path}")


def parquet_batch_generator(
    path: str,
    feature_cols: List[str],
    batch_size: int,
    repeat: bool = True,
    read_batch_size: Optional[int] = None,
):
    """
    Gera batches de (X, y) a partir de um arquivo parquet,
    sem carregar tudo em memória.

    Args:
        path: Caminho do parquet.
        feature_cols: Lista de colunas de features.
        batch_size: Tamanho do batch entregue ao modelo.
        repeat: Se True, repete infinitamente (treino). Se False, uma passagem (validação).
        read_batch_size: Linhas lidas do disco por vez (default: batch_size). Maior = menos I/O, mais RAM.
    """
    if read_batch_size is None:
        read_batch_size = batch_size

    def _iterate():
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=read_batch_size):
            df_batch = batch.to_pandas()
            if 'n' not in df_batch.columns:
                raise ValueError("DataFrame deve ter coluna 'n' (target)")
            X_batch = df_batch[feature_cols].values.astype(np.float32)
            y_batch = df_batch['n'].values.astype(np.float32)
            del df_batch
            gc.collect()
            # Se leu mais que um batch, entrega em fatias do tamanho do modelo
            n = X_batch.shape[0]
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                yield X_batch[start:end], y_batch[start:end]

    if repeat:
        while True:
            yield from _iterate()
    else:
        yield from _iterate()


class _ValDataWrapper:
    """
    Wrapper para validation_data: a cada época o Keras chama iter() e precisa
    de um gerador novo (o gerador com repeat=False se esgota numa época).
    """
    def __init__(self, val_path: str, feature_cols: List[str], batch_size: int):
        self.val_path = val_path
        self.feature_cols = feature_cols
        self.batch_size = batch_size

    def __iter__(self):
        return parquet_batch_generator(
            self.val_path, self.feature_cols, self.batch_size, repeat=False
        )


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
    batch_size: int = 65536,
    validation_split: float = 0.15,
    patience: int = 15
) -> pd.DataFrame:
    """
    Avalia rede neural densa usando os vizinhos mais próximos.
    
    Usa features estruturadas (ponderadas por distância e altitude).
    Usa AdamW com weight decay para regularização.
    
    base_path deve estar em data_dense (ex: data/data_dense/pressure/pressure).
    Treino/val/teste: {base_path}_train.parquet, _val.parquet, _test.parquet.
    Se _val não existir, gera a partir de data/data_train/{...}/_train.parquet (batches), libera memória, depois treina.
    Sempre usa monitor='val_loss'.
    
    Args:
        base_path: Caminho base em data_dense (ex: 'data/data_dense/pressure/pressure')
        output_dir: Diretório de saída para métricas
        models_dir: Diretório para salvar modelos
        variable_name: Nome da variável para o CSV (extraído do base_path se None)
        n_neighbors: Número de vizinhos a usar (default: 20)
        learning_rate: Taxa de aprendizado (default: 0.001)
        weight_decay: Regularização L2 via AdamW (default: 0.01)
        epochs: Número máximo de épocas
        batch_size: Tamanho do batch (default: 65536; 62GB suporta)
        validation_split: Fração usada ao gerar _val.parquet a partir do train (default: 0.15)
        patience: Épocas sem melhora antes de parar (default: 15)
    
    Returns:
        DataFrame com os resultados das métricas
    """
    train_path = f"{base_path}_train.parquet"
    test_path = f"{base_path}_test.parquet"
    # Se train/test não existirem em data_dense, usa data_train (mesma origem do val)
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
    # _val sempre em train/dense/data_dense/{var}/{var}_val.parquet (mesmo diretório do script)
    data_dense_output_base = os.path.join(SCRIPT_DIR, 'data_dense', variable_name, variable_name)
    val_path = f"{data_dense_output_base}_val.parquet"
    print(f"Variável: {variable_name}")

    # Garante *_val.parquet em data_dense: se não existir, gera a partir de
    # data/data_train/{...}/{...}_train.parquet (batches), libera memória, depois treina.
    ensure_val_parquet(val_path, base_path, validation_fraction=validation_split)

    if not os.path.isfile(train_path):
        raise FileNotFoundError(
            f"Arquivo de treino não encontrado: {train_path} "
            "(nem em data_dense nem em data_train)"
        )
    print(f"Lendo treino: {train_path}")
    feature_names, n_train_rows = infer_feature_columns_from_parquet(
        train_path,
        n_neighbors=n_neighbors
    )
    print(f"  → {n_train_rows:,} registros (lidos em batches)")
    print(f"  → {len(feature_names)} features")

    _, n_val_rows = infer_feature_columns_from_parquet(val_path, n_neighbors=n_neighbors)
    validation_steps = int(np.ceil(n_val_rows / batch_size))
    print(f"  → Validação: {val_path} ({n_val_rows:,} registros, em batches)")
    
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
    print(f"  → Monitor: val_loss")
    print(f"  → Early stopping: {patience} épocas sem melhora")
    print()

    steps_per_epoch = int(np.ceil(n_train_rows / batch_size))
    # tf.data com prefetch: carrega próximo batch enquanto treina (usa mais CPU, mais rápido)
    n_features = len(feature_names)
    read_batch_size = min(2_000_000, (n_train_rows // 10) or batch_size)  # blocos grandes = mais RAM
    train_ds = tf.data.Dataset.from_generator(
        lambda: parquet_batch_generator(
            train_path, feature_names, batch_size, read_batch_size=read_batch_size
        ),
        output_signature=(
            tf.TensorSpec(shape=(None, n_features), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    )
    train_ds = train_ds.prefetch(12)  # buffer grande = mais RAM, menos espera por I/O

    # Validação como tf.data.Dataset (Keras 3 não aceita _ValDataWrapper)
    val_ds = tf.data.Dataset.from_generator(
        lambda: parquet_batch_generator(
            val_path, feature_names, batch_size, repeat=False
        ),
        output_signature=(
            tf.TensorSpec(shape=(None, n_features), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    )

    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Carrega melhor modelo
    print(f"\nCarregando melhor modelo de: {model_path}")
    model = load_model(model_path)
    
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

# base_path='data/data_dense/temperature/temperature'
# base_path='data/data_dense/humidity/humidity'
# base_path='data/data_dense/radiation/radiation'
# base_path='data/data_dense/pressure/pressure'
# base_path='data/data_dense/rainfall/rainfall'

if __name__ == '__main__':
    evaluate_dense(
        base_path='../../data/data_dense/pressure/pressure',
        output_dir='train/dense/results',
        models_dir='train/dense/models',
        learning_rate=0.001,
        weight_decay=0.01,
        epochs=100,
        batch_size=65536,
        validation_split=0.15,
        patience=15
    )
