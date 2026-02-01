"""
Adiciona os K vizinhos mais próximos (com dado na mesma hora) a cada registro.

Salva ano a ano em data_train/<var>/anos/<year>.parquet.
YEARS_TO_PROCESS vazio = todos os anos; com valores = só esses anos.
Depois: pipenv run python scripts/join_neighbors.py para juntar em um único parquet.

pipenv run python scripts/prepare_train_data/add_neighbors.py
"""
import pandas as pd
from pathlib import Path

# Raiz do projeto (scripts/prepare_train_data/add_neighbors.py → sobe 3 níveis)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_TRAIN_DIR = BASE_DIR / "data_train"
DISTANCES_PATH = DATA_DIR / "station_distances.parquet"
K = 20

VARIABLES = ["pressure"]  # ,temperature, "humidity", "radiation", "pressure", "rainfall"

# Anos a processar. Vazio = todos. Ex: [2000, 2001] = só 2000 e 2001.
YEARS_TO_PROCESS = []


def main():
    print("=" * 80)
    print("ADICIONAR VIZINHOS POR (CODE, TIME) — processamento ano a ano")
    print("=" * 80)
    if YEARS_TO_PROCESS:
        print(f"   📌 Anos: {YEARS_TO_PROCESS}")
    else:
        print("   📌 Anos: todos (YEARS_TO_PROCESS vazio)")

    if not DISTANCES_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {DISTANCES_PATH}")

    print(f"\n📂 Carregando {DISTANCES_PATH}...")
    dist = pd.read_parquet(DISTANCES_PATH)
    dist = dist.sort_values(["stationA", "distancia_km"])
    print(f"   ✓ {len(dist):,} pares (stationA, stationB)")

    neighbors_by_a = {}
    for a, g in dist.groupby("stationA"):
        rows = g[["stationB", "distancia_km", "dif_altura_km"]].values.tolist()
        neighbors_by_a[a] = [
            (b, float(d), None if pd.isna(da) else float(da)) for b, d, da in rows
        ]

    for VARIABLE in VARIABLES:
        print(f"\n🔍 Processando {VARIABLE}...")

        var_dir = DATA_TRAIN_DIR / VARIABLE
        anos_dir = var_dir / "anos"
        input_path = var_dir / f"{VARIABLE}.parquet"

        if not input_path.exists():
            print(f"   ⏭️  {input_path} não encontrado, pulando.")
            continue

        df_time = pd.read_parquet(input_path, columns=["time"])
        df_time["time"] = pd.to_datetime(df_time["time"])
        all_years = sorted(df_time["time"].dt.year.unique())
        del df_time

        if YEARS_TO_PROCESS:
            years = sorted(y for y in YEARS_TO_PROCESS if y in all_years)
            skipped = set(YEARS_TO_PROCESS) - set(years)
            if skipped:
                print(f"   ⚠️ Anos não encontrados nos dados (pulados): {skipped}")
        else:
            years = all_years

        if not years:
            print("   ⏭️  Nenhum ano a processar, pulando.")
            continue

        base_cols = ["code", "time", VARIABLE]
        neighbor_cols = []
        for i in range(1, K + 1):
            neighbor_cols.extend([f"n{i}_code", f"n{i}_{VARIABLE}", f"n{i}_dist_km", f"n{i}_altdiff_km"])

        anos_dir.mkdir(parents=True, exist_ok=True)
        total_no = 0
        total_partial = 0

        for year in years:
            print(f"\n   📅 Executando ano {year}...")
            ts_lo = pd.Timestamp(f"{year}-01-01")
            ts_hi = pd.Timestamp(f"{year + 1}-01-01")
            filters = [("time", ">=", ts_lo), ("time", "<", ts_hi)]
            df = pd.read_parquet(input_path, filters=filters)
            df["time"] = pd.to_datetime(df["time"])

            by_code_time = df.set_index(["code", "time"])[VARIABLE].to_dict()
            available = set(by_code_time.keys())

            rows = []
            n_no_neighbors = 0
            n_partial = 0

            for _, r in df.iterrows():
                code, t, val = r["code"], r["time"], r[VARIABLE]
                row = {"code": code, "time": t, VARIABLE: val}
                cand = neighbors_by_a.get(code, [])
                valid = [(b, d, da) for b, d, da in cand if (b, t) in available]
                top = valid[:K]

                for i in range(K):
                    prefix = f"n{i+1}_"
                    row[prefix + "code"] = top[i][0] if i < len(top) else None
                    row[prefix + VARIABLE] = by_code_time.get((top[i][0], t)) if i < len(top) else None
                    row[prefix + "dist_km"] = top[i][1] if i < len(top) else None
                    row[prefix + "altdiff_km"] = top[i][2] if i < len(top) else None

                if len(top) == 0:
                    n_no_neighbors += 1
                elif len(top) < K:
                    n_partial += 1
                rows.append(row)

            out_year = pd.DataFrame(rows, columns=base_cols + neighbor_cols)
            for i in range(1, K + 1):
                for c in [f"n{i}_{VARIABLE}", f"n{i}_dist_km", f"n{i}_altdiff_km"]:
                    out_year[c] = pd.to_numeric(out_year[c], errors="coerce")

            out_path = anos_dir / f"{year}.parquet"
            out_year.to_parquet(out_path, index=False, compression="snappy")
            total_no += n_no_neighbors
            total_partial += n_partial
            print(f"      ✓ {len(out_year):,} registros → {out_path.name}")

        print(f"\n   ✓ Total | Sem vizinhos: {total_no:,} | Com <{K} vizinhos: {total_partial:,}")
        print(f"   💾 Arquivos em {anos_dir}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        raise
