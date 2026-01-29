"""
Script para preparar dados de treino a partir de weather_measurements.parquet.

- Lê data/weather_measurements.parquet
- radiation: valores negativos → 0
- Normaliza o DF (temperature, humidity, rainfall, radiation, pressure) e salva
  o scaler em data_train/
- Para cada coluna (temperature, humidity, rainfall, radiation, pressure):
  - Agrupa por code e (data + hora)
  - Remove dados faltantes daquela coluna
  - Gera DF com code, time, <nome da coluna>
  - Salva em ordem crescente em data_rain/<nome da coluna>/

pipenv run python scripts/prepare_train_data/prepare_train_data.py
"""
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

VARIABLES = ['temperature', 'humidity', 'rainfall', 'radiation', 'pressure']
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
INPUT_PATH = DATA_DIR / 'weather_measurements.parquet'
DATA_TRAIN_DIR = BASE_DIR / 'data_train'
SCALER_PATH = DATA_TRAIN_DIR / 'scaler.joblib'


def main():
    print("=" * 80)
    print("PREPARAÇÃO DE DADOS PARA TREINO")
    print("=" * 80)

    print(f"\n📂 Carregando {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    df['measurement_time'] = pd.to_datetime(df['measurement_time'])
    df.loc[df['radiation'] < 0, 'radiation'] = 0

    # 2. Normalizar (StandardScaler) e salvar scaler em data_train
    DATA_TRAIN_DIR.mkdir(exist_ok=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[VARIABLES] = scaler.fit_transform(df[VARIABLES])
    dump(scaler, SCALER_PATH)
    print(f"\n📐 DF normalizado (StandardScaler). Scaler salvo em {SCALER_PATH}")

    # 3. Agrupar por data+hora e gerar arquivos por coluna
    df['time'] = df['measurement_time'].dt.floor('h')

    for col in VARIABLES:
        out_dir = DATA_TRAIN_DIR / col
        out_dir.mkdir(parents=True, exist_ok=True)

        sub = df[['code', 'time', col]].copy()
        sub = sub.dropna(subset=[col])
        sub = sub.groupby(['code', 'time'], as_index=False)[col].mean()
        sub = sub.sort_values(['time', 'code'], ascending=True).reset_index(drop=True)

        out_path = out_dir / f"{col}.parquet"
        sub.to_parquet(out_path, index=False, compression='snappy')
        print(f"   💾 {col}: {len(sub):,} linhas → {out_path}")

    print("\n" + "=" * 80)
    print("✓ Concluído.")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        raise
