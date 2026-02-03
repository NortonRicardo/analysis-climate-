"""
Script simples para calcular distâncias e diferenças de altitude entre estações.

Para cada estação, calcula distância e diferença de altitude para todas as outras estações.

# station | neighbor | distance_km | rank | dif_altitude

pipenv run python scripts/calculate_distances.py
"""
import numpy as np
import pandas as pd
path = 'data/station.parquet'

df = pd.read_parquet(path)

# --- Preparação: garantir colunas necessárias e remover linhas com lat/lon faltando
df_coords = df[['code', 'latitude', 'longitude', 'altitude']].dropna().reset_index(drop=True)

# converter para radianos
coords_rad = np.radians(df_coords[['latitude', 'longitude']].to_numpy())

lat = coords_rad[:, 0][:, None]   # shape (n,1)
lon = coords_rad[:, 1][:, None]   # shape (n,1)

# diferenças por broadcasting
dlat = lat - lat.T                # shape (n,n)
dlon = lon - lon.T

# Haversine vetorizado
a = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2
c = 2 * np.arcsin(np.sqrt(a))
R = 6371.0  # raio da Terra em km
dist_matrix = R * c              # shape (n,n), distância em km

# remover pares i == j (distância zero)
n = dist_matrix.shape[0]
mask = ~np.eye(n, dtype=bool)    # True quando i != j
row_idx, col_idx = np.where(mask)

# construir DataFrame de pares
pairs = pd.DataFrame({
    'station': df_coords['code'].values[row_idx],
    'neighbor': df_coords['code'].values[col_idx],
    'distance_km': dist_matrix[row_idx, col_idx],
    'dif_altitude': df_coords['altitude'].values[row_idx] - df_coords['altitude'].values[col_idx]
})

# arredondar e ordenar e atribuir rank por estação
pairs['distance_km'] = pairs['distance_km'].round(1)
pairs = pairs.sort_values(['station', 'distance_km']).reset_index(drop=True)
pairs['rank'] = pairs.groupby('station').cumcount() + 1

pairs.to_parquet('data/train/station_neighbors.parquet', index=False)