"""
Lista vizinhos mais próximos de cada estação por variável.

Para cada estação, lista os vizinhos dentro do limite de distância específico
de cada variável climática.

pipenv run python scripts/list_neighbors.py
"""
import pandas as pd
from pathlib import Path
import json

# Configurações
DATA_DIR = Path(__file__).parent.parent / 'data'
STATION_DISTANCES_PATH = DATA_DIR / 'station_distances.parquet'
OUTPUT_JSON = DATA_DIR / 'station_neighbors.json'

# Limites de distância por variável (em km)
VARIABLES_LIMITS = {
    'temperature': 150,
    'humidity': 100,
    'rainfall': 50,
    'radiation': 255,
    'pressure': 300
}


def main():
    """Função principal"""
    print("=" * 80)
    print("LISTAGEM DE VIZINHOS POR ESTAÇÃO E VARIÁVEL")
    print("=" * 80)
    
    # Carrega distâncias
    print(f"\n📊 Carregando distâncias de {STATION_DISTANCES_PATH}...")
    if not STATION_DISTANCES_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {STATION_DISTANCES_PATH}")
    
    df_distances = pd.read_parquet(STATION_DISTANCES_PATH)
    print(f"  ✓ {len(df_distances):,} distâncias carregadas")
    
    # Obtém lista única de estações
    all_stations = sorted(set(df_distances['stationA'].unique()) | set(df_distances['stationB'].unique()))
    print(f"  ✓ {len(all_stations)} estações únicas")
    
    # Estrutura para armazenar resultados
    results = []
    
    print(f"\n🔍 Processando estações...")
    
    for idx, station_code in enumerate(all_stations):
        station_data = {'code': station_code}
        
        # Para cada variável, busca vizinhos dentro do limite
        for variable, max_distance in VARIABLES_LIMITS.items():
            # Busca vizinhos onde station_code é stationA
            neighbors = df_distances[
                (df_distances['stationA'] == station_code) &
                (df_distances['distancia_km'] <= max_distance)
            ].copy()
            
            # Ordena do mais próximo para o mais distante
            neighbors = neighbors.sort_values('distancia_km', ascending=True)
            
            # Cria lista de vizinhos no formato solicitado
            neighbors_list = [
                {'code': row['stationB'], 'distancia': round(row['distancia_km'], 2)}
                for _, row in neighbors.iterrows()
            ]
            
            station_data[variable] = neighbors_list
        
        results.append(station_data)
        
        # Progresso
        if (idx + 1) % 100 == 0:
            print(f"  Processando... {idx + 1}/{len(all_stations)}")
    
    # Salva JSON (estrutura aninhada - mais rápido para busca)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    file_size = OUTPUT_JSON.stat().st_size / (1024 * 1024)  # MB
    print(f"\n💾 JSON salvo em: {OUTPUT_JSON}")
    print(f"  ✓ {len(results)} estações processadas")
    print(f"  ✓ Tamanho: {file_size:.2f} MB")
    print(f"  ✓ Estrutura: results[station_code]['variable'] - acesso O(1)")
    
    # Estatísticas
    print(f"\n📊 Estatísticas:")
    for variable in VARIABLES_LIMITS.keys():
        total_neighbors = sum(len(station.get(variable, [])) for station in results)
        avg_neighbors = total_neighbors / len(results) if len(results) > 0 else 0
        print(f"  - {variable}: média de {avg_neighbors:.1f} vizinhos por estação")
    
    print("\n✓ CONCLUÍDO!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
