"""
Módulo para avaliar regressão linear usando os 20 vizinhos mais próximos.

Usa os valores dos vizinhos (n1 a n20), suas distâncias e diferenças de altitude
como features para prever o valor observado.

Usa arquivos separados de treino (_train.parquet) e teste (_test.parquet).

Os resultados são salvos em CSV e podem ser acumulados para múltiplas variáveis.

pipenv run python train/linear_regression/linear_regression.py
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, List


# =============================================================================
# MÉTRICAS (implementação manual, sem dependências externas)
# =============================================================================

def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error (Erro Médio Absoluto)
    
    MAE = (1/n) * Σ|y_true - y_pred|
    """
    return np.mean(np.abs(y_true - y_pred))


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error (Raiz do Erro Quadrático Médio)
    
    RMSE = sqrt((1/n) * Σ(y_true - y_pred)²)
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calc_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Bias (Viés/Tendência)
    
    Bias = (1/n) * Σ(y_pred - y_true)
    """
    return np.mean(y_pred - y_true)


def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coeficiente de Determinação (R²)
    
    R² = 1 - (SS_res / SS_tot)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return np.nan
    
    return 1 - (ss_res / ss_tot)


def calc_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coeficiente de Correlação de Pearson (r)
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
# REGRESSÃO LINEAR (implementação manual via OLS - Mínimos Quadrados Ordinários)
# =============================================================================

class LinearRegression:
    """
    Regressão Linear via Mínimos Quadrados Ordinários (OLS)
    
    Fórmula: β = (X'X)^(-1) X'y
    
    Onde:
        X = matriz de features (com coluna de 1s para intercepto)
        y = vetor de valores alvo
        β = vetor de coeficientes [intercepto, coef1, coef2, ...]
    """
    
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.feature_names = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """
        Treina o modelo de regressão linear.
        
        Args:
            X: Matriz de features (n_samples, n_features)
            y: Vetor alvo (n_samples,)
            feature_names: Nomes das features (opcional)
        """
        n_samples, n_features = X.shape
        
        # Adiciona coluna de 1s para o intercepto
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        
        # Calcula os coeficientes via OLS: β = (X'X)^(-1) X'y
        # Usando pseudo-inversa para maior estabilidade numérica
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        
        try:
            # Tenta inversa direta primeiro
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Se falhar, usa pseudo-inversa
            beta = np.linalg.pinv(XtX) @ Xty
        
        self.intercept = beta[0]
        self.coefficients = beta[1:]
        self.feature_names = feature_names
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz predições usando o modelo treinado.
        
        Args:
            X: Matriz de features (n_samples, n_features)
            
        Returns:
            Vetor de predições (n_samples,)
        """
        if self.coefficients is None:
            raise ValueError("Modelo não treinado. Execute fit() primeiro.")
        
        return self.intercept + X @ self.coefficients
    
    def get_coefficients_df(self) -> pd.DataFrame:
        """
        Retorna DataFrame com os coeficientes e seus nomes.
        """
        if self.coefficients is None:
            return None
        
        names = self.feature_names if self.feature_names else [f'x{i}' for i in range(len(self.coefficients))]
        
        return pd.DataFrame({
            'feature': ['intercept'] + list(names),
            'coefficient': [self.intercept] + list(self.coefficients)
        })


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def detect_variable_name(df: pd.DataFrame) -> str:
    """
    Detecta automaticamente o nome da variável no DataFrame.
    """
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
    
    Features:
        - n1_{var} até n{n_neighbors}_{var}: valores dos vizinhos
        - n1_dist_km até n{n_neighbors}_dist_km: distâncias
        - n1_altdiff_km até n{n_neighbors}_altdiff_km: diferenças de altitude
    
    Returns:
        Tuple com (X, y, feature_names)
    """
    feature_cols = []
    
    # Colunas dos valores dos vizinhos
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
    
    # Target (valor observado)
    target_col = var_name
    
    # Seleciona colunas necessárias
    all_cols = [target_col] + feature_cols
    data = df[all_cols].copy()
    
    # Remove linhas com NaN
    data_valid = data.dropna()
    
    y = data_valid[target_col].values
    X = data_valid[feature_cols].values
    
    return X, y, feature_cols


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula todas as métricas de avaliação.
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

def evaluate_linear_regression(
    base_path: str,
    output_dir: str = 'train/linear_regression/results',
    variable_name: Optional[str] = None,
    n_neighbors: int = 20,
    save_coefficients: bool = True
) -> pd.DataFrame:
    """
    Avalia regressão linear usando os vizinhos mais próximos.
    
    Usa arquivos separados de treino e teste:
        - {base_path}_train.parquet
        - {base_path}_test.parquet
    
    Args:
        base_path: Caminho base (ex: 'data_train/temperature/temperature')
                   Vai carregar {base_path}_train.parquet e {base_path}_test.parquet
        output_dir: Diretório de saída
        variable_name: Nome da variável (auto-detectado se None)
        n_neighbors: Número de vizinhos a usar (default: 20)
        save_coefficients: Se deve salvar os coeficientes em arquivo separado
    
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
    print(f"  → {len(y_train):,} amostras válidas de treino")
    print(f"  → {len(feature_names)} features: {n_neighbors} vizinhos × 3 (valor, dist, alt)")
    
    # Prepara features de teste
    X_test, y_test, _ = prepare_features(df_test, variable_name, n_neighbors)
    print(f"  → {len(y_test):,} amostras válidas de teste")
    
    # Treina modelo
    print("\nTreinando regressão linear...")
    model = LinearRegression()
    model.fit(X_train, y_train, feature_names)
    
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
            'n_samples', 'mae', 'rmse', 'bias', 'r', 'r2',
            'mean_observed', 'mean_predicted', 'std_observed', 'std_predicted']
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
        coef_df = model.get_coefficients_df()
        coef_file = os.path.join(output_dir, f'coefficients_{variable_name}.csv')
        coef_df.to_csv(coef_file, index=False)
        print(f"✓ Coeficientes salvos em: {coef_file}")
    
    # Mostra resultados
    print(f"\n{'='*60}")
    print(f"REGRESSÃO LINEAR PARA: {variable_name}")
    print(f"{'='*60}")
    print(f"  Vizinhos usados:   {n_neighbors}")
    print(f"  Features:          {len(feature_names)}")
    print(f"  Amostras treino:   {len(y_train):,}")
    print(f"  Amostras teste:    {len(y_test):,}")
    print(f"  ---")
    print(f"  MAE:               {metrics['mae']:.4f}")
    print(f"  RMSE:              {metrics['rmse']:.4f}")
    print(f"  Bias:              {metrics['bias']:.4f}")
    print(f"  r (correlação):    {metrics['r']:.4f}")
    print(f"  R²:                {metrics['r2']:.4f}")
    print(f"{'='*60}")
    
    # Mostra top 10 coeficientes mais importantes (por valor absoluto)
    coef_df = model.get_coefficients_df()
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
