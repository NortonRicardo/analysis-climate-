"""
Script para LIMPAR e separar variáveis climáticas do arquivo weather_measurements.

Este script realiza LIMPEZA dos dados:
- Para cada variável (temperature, humidity, rainfall, radiation, pressure):
  - Mantém apenas code, measurement_time e a variável específica
  - REMOVE registros com dados faltantes para aquela variável específica
  - Para radiation: valores < 0 são CORRIGIDOS para 0
- Salva dados LIMPOS em arquivos separados dentro de data/weather_cleaned/

pipenv run python data_download/clean_weather_variables.py
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

# Variáveis a processar
VARIABLES = ['temperature', 'humidity', 'rainfall', 'radiation', 'pressure']

# Colunas base que sempre serão mantidas
BASE_COLUMNS = ['code', 'measurement_time']

def clean_weather_variables():
    """
    Limpa e separa as variáveis climáticas em arquivos individuais.
    
    Realiza limpeza dos dados removendo registros com dados faltantes
    e corrigindo valores inválidos.
    """
    
    # Caminhos dos arquivos
    data_dir = Path(__file__).parent.parent / 'data'
    input_path = data_dir / 'weather_measurements.parquet'
    output_dir = data_dir / 'weather_cleaned'
    
    # Verifica se o arquivo de entrada existe
    if not input_path.exists():
        raise FileNotFoundError(
            f"Arquivo {input_path} não encontrado.\n"
            "Execute primeiro: pipenv run python data_download/download_weather_measurements.py"
        )
    
    print("=" * 80)
    print("LIMPEZA E PROCESSAMENTO DE VARIÁVEIS CLIMÁTICAS")
    print("=" * 80)
    print("Este script realiza LIMPEZA dos dados:")
    print("  - Remove registros com dados faltantes por variável")
    print("  - Corrige valores inválidos (radiation < 0)")
    print("  - Gera arquivos limpos e separados por variável")
    print("=" * 80)
    
    # Cria diretório de saída
    output_dir.mkdir(exist_ok=True)
    
    # Carrega o arquivo de medições
    print(f"\nCarregando dados de {input_path}...")
    print("  (Isso pode levar alguns minutos para arquivos grandes...)")
    
    df = pd.read_parquet(input_path)
    
    print(f"✓ {len(df):,} registros carregados")
    print(f"✓ Colunas disponíveis: {', '.join(df.columns)}")
    
    # Verifica se as variáveis existem no arquivo
    missing_vars = [var for var in VARIABLES if var not in df.columns]
    if missing_vars:
        raise ValueError(f"Variáveis não encontradas no arquivo: {missing_vars}")
    
    # Processa cada variável
    print(f"\n🧹 Limpando e processando {len(VARIABLES)} variáveis...\n")
    
    for variable in VARIABLES:
        print(f"🧹 Limpando {variable}...")
        
        # Seleciona apenas as colunas necessárias
        columns_to_keep = BASE_COLUMNS + [variable]
        df_var = df[columns_to_keep].copy()
        
        # Tratamento especial para radiation
        if variable == 'radiation':
            # Valores < 0 recebem 0 (correção de dados inválidos)
            negative_count = (df_var[variable] < 0).sum()
            if negative_count > 0:
                print(f"  🔧 {negative_count:,} valores negativos CORRIGIDOS para 0")
                df_var.loc[df_var[variable] < 0, variable] = 0
        
        # Remove registros com dados faltantes para esta variável específica (LIMPEZA)
        initial_count = len(df_var)
        df_var = df_var.dropna(subset=[variable])
        removed_count = initial_count - len(df_var)
        
        if removed_count > 0:
            print(f"  🗑️  {removed_count:,} registros com dados faltantes REMOVIDOS (limpeza)")
        else:
            print(f"  ✓ Nenhum dado faltante encontrado")
        
        # Ordena por measurement_time (já deve estar ordenado, mas garantimos)
        df_var['measurement_time'] = pd.to_datetime(df_var['measurement_time'])
        df_var = df_var.sort_values('measurement_time', ascending=True)
        df_var = df_var.reset_index(drop=True)
        
        # Salva o arquivo limpo
        output_path = output_dir / f'weather_{variable}_cleaned.parquet'
        df_var.to_parquet(output_path, index=False, compression='snappy')
        
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        print(f"  💾 Arquivo LIMPO salvo: {output_path.name}")
        print(f"  ✓ Registros finais (após limpeza): {len(df_var):,}")
        print(f"  ✓ Tamanho: {file_size:.2f} MB")
        
        # Estatísticas da variável
        if len(df_var) > 0:
            stats = df_var[variable].describe()
            print(f"  📈 Estatísticas:")
            print(f"     - Mínimo: {stats['min']:.2f}")
            print(f"     - Máximo: {stats['max']:.2f}")
            print(f"     - Média: {stats['mean']:.2f}")
            print(f"     - Mediana: {stats['50%']:.2f}")
        
        print()
    
    print("=" * 80)
    print("✓ LIMPEZA E PROCESSAMENTO CONCLUÍDOS!")
    print("=" * 80)
    print(f"\n📁 Arquivos LIMPOS salvos em: {output_dir}")
    print("\n📄 Arquivos gerados (dados limpos):")
    for variable in VARIABLES:
        file_path = output_dir / f'weather_{variable}_cleaned.parquet'
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)
            print(f"  - weather_{variable}_cleaned.parquet ({size:.2f} MB)")

if __name__ == "__main__":
    try:
        clean_weather_variables()
    except FileNotFoundError as e:
        print(f"❌ Erro: {e}")
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()

