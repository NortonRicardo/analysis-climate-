"""
Script para preparar dados de treino a partir de weather_measurements.parquet.

- Lê data/weather_measurements.parquet
- radiation: valores negativos → 0
- Para cada variável (temperature, humidity, rainfall, radiation, pressure):
  - Separa coluna com code e data/hora (measurement_time)
  - Remove duplicados
  - Ordena por data/hora
  - Salva em data_train/{variavel}/measurement.parquet

pipenv run python scripts/2.3_prepare_train_data.py
"""

import pandas as pd
import numpy as np
import gc
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, Parallel, delayed

VARIABLES = ['humidity']  # 'temperature','humidity', 'rainfall', 'pressure', 'radiation'
YEAR_START = 2000
YEAR_END = 2025
N_JOBS = 5  # usa todos os cores; ajuste (ex: 7) se quiser deixar 1 livre

# Paths pensados para rodar em ambiente como Google Colab,
# assumindo estrutura:
#   data/weather_measurements.parquet
#   data/station_neighbors.parquet
#   data/data_train/
INPUT_PATH = Path('data/weather_measurements.parquet')
DATA_TRAIN_DIR = Path('data/data_train')
DATA_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

def stage_1():
  print("=" * 80)
  print("PREPARAÇÃO DE DADOS PARA TREINO")
  print("=" * 80)

  DATA_TRAIN_DIR.mkdir(exist_ok=True)
  print(f"\n Separacao de dados por variaveis climaticas...")

  for col in VARIABLES:
      print(f"   🔍 Separacao de dados para {col}...")
      print(f"   📂 Carregando apenas colunas necessárias de {INPUT_PATH}...")
      df = pd.read_parquet(INPUT_PATH, columns=['code', 'measurement_time', col])
      df['measurement_time'] = pd.to_datetime(df['measurement_time'])
      if col == 'radiation':
          df.loc[df[col] < 0, col] = 0

      out_dir = DATA_TRAIN_DIR / col
      out_dir.mkdir(parents=True, exist_ok=True)

      print(f"   🔍 Remocao de dados faltantes...")
      df = df.dropna(subset=[col])

      print(f"   🔍 Remocao de duplicados...")
      df = df.drop_duplicates()

      print(f"   🔍 Ordenacao de dados...")
      df = df.sort_values(['measurement_time', 'code'], ascending=True).reset_index(drop=True)

      print(f"   🔍 Salvao de dados...")
      out_path = out_dir / 'measurement.parquet'
      df.to_parquet(out_path, index=False, compression='snappy')
      print(f"   💾 {col}: {len(df):,} linhas → {out_path}")

      del df
      gc.collect()

  print("\n" + "=" * 80)
  print("✓ STAGE 1 CONCLUIDO!!!")
  print("=" * 80)


def _process_group(group, col, dist_by_station, eps):
  """
  Processa um grupo (um measurement_time): para cada estação no grupo,
  monta a linha com os 20 vizinhos mais próximos que têm medição nesse instante.
  Usado em paralelo por joblib.
  """
  codes_in_group = set(group['code'])
  group_dict = {r.code: r for r in group.itertuples(index=False)}
  rows_list = []

  for row in group.itertuples(index=False):
    code = row.code
    other_codes = codes_in_group - {code}
    if not other_codes:
      continue

    station_dists = dist_by_station.get(code)
    if station_dists is None:
      continue

    neighbors_all = station_dists.index
    ordered = [nb for nb in neighbors_all if nb in other_codes][:20]
    if not ordered:
      continue

    ordered_rows = [group_dict[nb] for nb in ordered if nb in group_dict]
    new_row = {
        'measurement_time': row.measurement_time,
        'code': code,
        'n': getattr(row, col),
    }

    for i in range(len(ordered_rows)):
      nb_row = ordered_rows[i]
      neighbor_code = nb_row.code
      dist_row = station_dists.loc[neighbor_code]
      val = getattr(nb_row, col)
      dist_km = dist_row['distance_km']
      new_row[f'n{i+1}'] = val
      new_row[f'n{i+1}_dist'] = dist_km
      new_row[f'n{i+1}_alt_diff'] = dist_row['dif_altitude'] / 1000.0
      new_row[f'n{i+1}_idw'] = val / (dist_km + eps)

    for i in range(len(ordered_rows), 20):
      new_row[f'n{i+1}'] = np.nan
      new_row[f'n{i+1}_dist'] = np.nan
      new_row[f'n{i+1}_alt_diff'] = np.nan
      new_row[f'n{i+1}_idw'] = np.nan

    rows_list.append(new_row)

  return rows_list

def stage_2():
  print("=" * 80)
  print("ADD NEIGHBORS (ano a ano + paralelo por measurement_time)")
  print("=" * 80)

  DIST_PATH = Path('data/station_neighbors.parquet')
  dist_df = pd.read_parquet(DIST_PATH)

  dist_by_station = {
      station: g.sort_values('rank').set_index('neighbor')
      for station, g in dist_df.groupby('station')
  }
  del dist_df
  gc.collect()

  eps = 1e-6

  for col in VARIABLES:
    print(f"   🔍 Processando {col}...")
    out_path = DATA_TRAIN_DIR / col / 'measurement.parquet'
    years = list(range(YEAR_START, YEAR_END + 1))
    print(f"   📅 Anos a processar: {years[0]} a {years[-1]} ({len(years)} anos) — lendo um ano por vez do parquet")

    years_dir = DATA_TRAIN_DIR / col / 'years'
    years_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
      filters = [
          ('measurement_time', '>=', pd.Timestamp(f'{year}-01-01')),
          ('measurement_time', '<', pd.Timestamp(f'{year+1}-01-01')),
      ]
      df_year = pd.read_parquet(out_path, filters=filters)
      if len(df_year) == 0:
        continue
      df_year['measurement_time'] = pd.to_datetime(df_year['measurement_time'])
      print(f"   🔍 Processando {year}... ({len(df_year):,} linhas)")
      groups = [g for _, g in df_year.groupby('measurement_time')]
      results = Parallel(n_jobs=N_JOBS)(
          delayed(_process_group)(g, col, dist_by_station, eps) for g in groups
      )
      rows_list = [row for sublist in results for row in sublist]
      df_year_out = pd.DataFrame(rows_list)
      path_year = years_dir / f'measurement_with_neighbors_{year}.parquet'
      df_year_out.to_parquet(path_year, index=False, compression='snappy')
      print(f"      {year}: {len(df_year_out):,} linhas → {path_year.name}")
      del df_year, groups, df_year_out, results, rows_list
      gc.collect()

    # print(f"   🔗 Escrevendo arquivo final em streaming (um ano por vez)...")
    # out_final = DATA_TRAIN_DIR / col / 'measurement_with_neighbors.parquet'
    # writer = None
    # total_rows = 0
    # for year in years:
    #   print(f"Lendo arquivo do ano {year}")
    #   path_year = years_dir / f'measurement_with_neighbors_{year}.parquet'
    #   if not path_year.exists():
    #     continue
    #   table = pq.read_table(path_year)
    #   if writer is None:
    #     writer = pq.ParquetWriter(out_final, table.schema, compression='snappy')
    #   writer.write_table(table)
    #   total_rows += table.num_rows
    #   del table
    #   gc.collect()
    # if writer is not None:
    #   writer.close()
    #   print(f"   💾 {col}: {total_rows:,} linhas → {out_final}")
    # else:
    #   print(f"   ⚠️ Nenhum dado processado para {col}. Pulando.")
    # gc.collect()
      
def _split_key_for_batch(row_offset, batch_len, random_state):
  """Chave pseudoaleatória em [0, 1) por índice de linha (reproduzível)."""
  idx = np.arange(row_offset, row_offset + batch_len, dtype=np.uint64)
  # hash determinístico: (idx * primo + seed) % 2^32 → [0, 1)
  key = (idx * np.uint64(2654435761) + np.uint64(random_state & 0xFFFFFFFF)) % np.uint64(2**32)
  return key.astype(np.float64) / (2**32)


def stage_3(batch_size=500_000, shuffle=False, random_state=42):
  """batch_size: linhas por lote. shuffle: True=split aleatório (60/40), False=sequencial."""
  print("=" * 80)
  print("NORMALIZANDO DADOS E SEPARANDO POR TRAIN E TESTE (em lotes, 60% train / 40% test)")
  print("=" * 80)
  print(f"   Split: {'aleatório (shuffle=True)' if shuffle else 'sequencial (shuffle=False)'}")

  for col in VARIABLES:
    print(f"   🔍 Processando {col}...")
    out_path = DATA_TRAIN_DIR / col / 'measurement_with_neighbors.parquet'
    train_path = DATA_TRAIN_DIR / col / f'{col}_train.parquet'
    test_path = DATA_TRAIN_DIR / col / f'{col}_test.parquet'
    scaler_path = DATA_TRAIN_DIR / col / 'scaler.joblib'

    pf = pq.ParquetFile(out_path)
    total_rows = pf.metadata.num_rows
    train_boundary = int(0.6 * total_rows) if not shuffle else None
    if not shuffle:
      print(f"   📊 Total: {total_rows:,} | Train: 0–{train_boundary:,} | Test: {train_boundary:,}–{total_rows:,}")
    else:
      print(f"   📊 Total: {total_rows:,} | Train ~60% | Test ~40% (aleatório, seed={random_state})")

    # Colunas a normalizar (conhecidas do schema)
    schema_names = pf.schema_arrow.names
    cols_to_normalize = [c for c in schema_names if c not in ('measurement_time', 'code')]

    # ----- Pass 1: min/max só dos lotes de treino -----
    print(f"   🔍 Pass 1: calculando min/max no treino...")
    running_min = None
    running_max = None
    row_offset = 0

    for batch in pf.iter_batches(batch_size=batch_size):
      df = batch.to_pandas()
      n = len(df)
      if shuffle:
        df['_split_key'] = _split_key_for_batch(row_offset, n, random_state)
      else:
        df['_row_idx'] = np.arange(row_offset, row_offset + n, dtype=np.int64)
      row_offset += n
      df = df.dropna()
      if shuffle:
        train_part = df[df['_split_key'] < 0.6].drop(columns=['_split_key'])
      else:
        train_part = df[df['_row_idx'] < train_boundary].drop(columns=['_row_idx'])
      if len(train_part) == 0:
        del df, train_part
        gc.collect()
        continue
      arr = train_part[cols_to_normalize]
      if running_min is None:
        running_min = arr.min(axis=0).values
        running_max = arr.max(axis=0).values
      else:
        running_min = np.minimum(running_min, arr.min(axis=0).values)
        running_max = np.maximum(running_max, arr.max(axis=0).values)
      del df, train_part, arr
      gc.collect()

    # Monta o scaler a partir do min/max (equivalente a fit)
    dummy = np.array([running_min, running_max])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(dummy)
    dump(scaler, scaler_path)
    print(f"   💾 Scaler salvo: {scaler_path}")
    del running_min, running_max, dummy
    gc.collect()

    # ----- Pass 2: transformar e ir salvando em train/test -----
    print(f"   🔍 Pass 2: normalizando e salvando por lote...")
    pf = pq.ParquetFile(out_path)
    train_writer = None
    test_writer = None
    row_offset = 0
    train_count = 0
    test_count = 0

    for batch in pf.iter_batches(batch_size=batch_size):
      df = batch.to_pandas()
      n = len(df)
      if shuffle:
        df['_split_key'] = _split_key_for_batch(row_offset, n, random_state)
      else:
        df['_row_idx'] = np.arange(row_offset, row_offset + n, dtype=np.int64)
      row_offset += n
      df = df.dropna()
      if shuffle:
        train_part = df[df['_split_key'] < 0.6].drop(columns=['_split_key'])
        test_part = df[df['_split_key'] >= 0.6].drop(columns=['_split_key'])
      else:
        train_part = df[df['_row_idx'] < train_boundary].drop(columns=['_row_idx'])
        test_part = df[df['_row_idx'] >= train_boundary].drop(columns=['_row_idx'])
      del df
      if len(train_part) > 0:
        train_part[cols_to_normalize] = scaler.transform(train_part[cols_to_normalize].values).astype(np.float32)
        tab = pa.Table.from_pandas(train_part, preserve_index=False)
        if train_writer is None:
          train_writer = pq.ParquetWriter(train_path, tab.schema, compression='snappy')
        train_writer.write_table(tab)
        train_count += len(train_part)
        del tab
      if len(test_part) > 0:
        test_part[cols_to_normalize] = scaler.transform(test_part[cols_to_normalize].values).astype(np.float32)
        tab = pa.Table.from_pandas(test_part, preserve_index=False)
        if test_writer is None:
          test_writer = pq.ParquetWriter(test_path, tab.schema, compression='snappy')
        test_writer.write_table(tab)
        test_count += len(test_part)
        del tab
      del train_part, test_part
      gc.collect()

    if train_writer is not None:
      train_writer.close()
    if test_writer is not None:
      test_writer.close()
    print(f"   💾 Train: {train_count:,} | Test: {test_count:,}")
    gc.collect()
    
def main():
    # print('STAGE 1: PREPARAÇÃO DE DADOS PARA TREINO')
    # stage_1()
    print('STAGE 2: ADD NEIGHBORS')
    stage_2()
    # print('STAGE 3: NORMALIZANDO DADOS E SEPARANDO POR TRAIN E TESTE')
    # stage_3()
    # print('Finished!')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        raise