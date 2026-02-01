"""
Junta os parquets ano a ano (data_train/<var>/anos/*.parquet) em um único
<var>_with_neighbors.parquet por variável.

Executar depois de add_neighbors.py. Lê todos os .parquet em cada anos/,
concatena, ordena por time/code, remove linhas com NaN, divide em train/test e salva.
Nota: o join carrega todos os anos em memória para ordenar; se estourar, rode
add_neighbors com YEARS_TO_PROCESS em lotes e junte depois (ou use só os anos/).

pipenv run python scripts/prepare_train_data/join_neighbors.py
"""
import pandas as pd
from pathlib import Path

# Raiz do projeto (scripts/prepare_train_data/join_neighbors.py → sobe 2 níveis)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_TRAIN_DIR = BASE_DIR / "data_train"

# Mesmas variáveis do add_neighbors. Ajuste se precisar.
VARIABLES = ["pressure"]

# Proporção para treino (resto = test). Split é temporal (primeiros TRAIN_RATIO para train).
TRAIN_RATIO = 0.6


def main():
    print("=" * 80)
    print("JUNTAR VIZINHOS (anos → único parquet, limpar NaN, train/test)")
    print("=" * 80)

    for var in VARIABLES:
        anos_dir = DATA_TRAIN_DIR / var / "anos"
        if not anos_dir.exists():
            print(f"\n⏭️  {var}: {anos_dir} não existe, pulando.")
            continue

        files = sorted(anos_dir.glob("*.parquet"))
        if not files:
            print(f"\n⏭️  {var}: nenhum .parquet em {anos_dir}, pulando.")
            continue

        print(f"\n🔍 {var}: {len(files)} arquivos em {anos_dir}")
        out = None
        for i, f in enumerate(files, 1):
            print(f"   [{i}/{len(files)}] Carregando {f.name}...", flush=True)
            df = pd.read_parquet(f)
            if out is None:
                out = df
            else:
                out = pd.concat([out, df], ignore_index=True)
            del df  # libera o arquivo lido; só o acumulado (out) fica na memória
        out = out.sort_values(["time", "code"]).reset_index(drop=True)

        n_antes = len(out)
        out = out.dropna()
        n_removidos = n_antes - len(out)
        n_final = len(out)
        print(f"   📊 Registros com NaN removidos: {n_removidos:,} (antes: {n_antes:,} → final: {n_final:,})")

        if n_final == 0:
            print(f"   ⚠️ Nenhum registro restante, pulando salvamento.")
            continue

        # Split temporal: primeiros TRAIN_RATIO para train, resto para test
        idx = int(n_final * TRAIN_RATIO)
        train_df = out.iloc[:idx]
        test_df = out.iloc[idx:]
        print(f"   📂 Train: {len(train_df):,} | Test: {len(test_df):,} (ratio {TRAIN_RATIO})")

        var_dir = DATA_TRAIN_DIR / var
        out.to_parquet(var_dir / f"{var}_with_neighbors.parquet", index=False, compression="snappy")
        train_df.to_parquet(var_dir / f"{var}_train.parquet", index=False, compression="snappy")
        test_df.to_parquet(var_dir / f"{var}_test.parquet", index=False, compression="snappy")
        print(f"   ✓ Salvos: {var}_with_neighbors.parquet, {var}_train.parquet, {var}_test.parquet")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        raise
