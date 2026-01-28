"""
Script simples para calcular distâncias e diferenças de altitude entre estações.

Para cada estação, calcula distância e diferença de altitude para todas as outras estações.

pipenv run python scripts/calculate_distances.py
"""
import pandas as pd
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

# Configurações
DATA_DIR = Path(__file__).parent.parent / 'data'
STATIONS_PATH = DATA_DIR / 'station.parquet'
OUTPUT_PATH = DATA_DIR / 'station_distances.parquet'


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula a distância geográfica entre dois pontos usando a fórmula de Haversine.
    
    O que é:
    - A fórmula de Haversine calcula a distância do "grande círculo" entre dois pontos
      na superfície de uma esfera (como a Terra)
    - É a menor distância entre dois pontos na superfície esférica
    - Considera a curvatura da Terra (não assume superfície plana)
    - Precisão: boa para distâncias curtas a médias (até alguns milhares de km)
    
    Como o cálculo é feito:
    1. Converte coordenadas de graus para radianos
    2. Calcula diferenças de latitude (dlat) e longitude (dlon)
    3. Aplica a fórmula de Haversine:
       a = sin²(dlat/2) + cos(lat1) × cos(lat2) × sin²(dlon/2)
       c = 2 × atan2(√a, √(1-a))
    4. Multiplica pelo raio da Terra para obter distância em km
    
    Parâmetros:
        lat1, lon1: Latitude e longitude do primeiro ponto (em graus)
        lat2, lon2: Latitude e longitude do segundo ponto (em graus)
    
    Retorna:
        Distância em quilômetros (km)
    """
    R = 6371.0  # Raio da Terra em quilômetros
    
    # Converte coordenadas de graus para radianos
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
    
    # Calcula diferenças de latitude e longitude
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Fórmula de Haversine
    # a = sin²(dlat/2) + cos(lat1) × cos(lat2) × sin²(dlon/2)
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    # c = 2 × atan2(√a, √(1-a)) - distância angular em radianos
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Multiplica pelo raio da Terra para obter distância em km
    return R * c


def main():
    """Função principal"""
    print("=" * 80)
    print("CÁLCULO DE DISTÂNCIAS ENTRE ESTAÇÕES")
    print("=" * 80)
    
    # Carrega estações
    print(f"\n📊 Carregando estações de {STATIONS_PATH}...")
    df_stations = pd.read_parquet(STATIONS_PATH)
    print(f"  ✓ {len(df_stations)} estações carregadas")
    
    # Verifica colunas necessárias
    required_cols = ['code', 'latitude', 'longitude', 'altitude']
    missing = [c for c in required_cols if c not in df_stations.columns]
    if missing:
        raise ValueError(f"Colunas faltando: {missing}")
    
    # Remove estações sem coordenadas válidas
    df_stations = df_stations.dropna(subset=['latitude', 'longitude'])
    print(f"  ✓ {len(df_stations)} estações com coordenadas válidas")
    
    # Lista para armazenar resultados
    results = []
    
    # Para cada estação A, calcula distância para todas as estações B
    print(f"\n🔢 Calculando distâncias...")
    total = len(df_stations) * (len(df_stations) - 1)
    calculated = 0
    
    for i, stationA in df_stations.iterrows():
        codeA = stationA['code']
        latA = stationA['latitude']
        lonA = stationA['longitude']
        altA = stationA['altitude']
        
        for j, stationB in df_stations.iterrows():
            # Não calcula distância de uma estação para ela mesma
            if i == j:
                continue
            
            codeB = stationB['code']
            latB = stationB['latitude']
            lonB = stationB['longitude']
            altB = stationB['altitude']
            
            # Calcula distância geográfica (km)
            distancia = round(haversine_distance(latA, lonA, latB, lonB), 2)
            
            # Calcula diferença de altitude (km) - B - A
            if pd.notna(altA) and pd.notna(altB):
                dif_altura = round((altB - altA) / 1000, 3)  # Converte metros para km
            else:
                dif_altura = None
            
            results.append({
                'stationA': codeA,
                'stationB': codeB,
                'distancia_km': distancia,
                'dif_altura_km': dif_altura
            })
            
            calculated += 1
            if calculated % 10000 == 0:
                print(f"  Processando... {calculated:,} / {total:,}")
    
    # Cria DataFrame e salva
    df_results = pd.DataFrame(results)
    
    print(f"\n💾 Salvando resultados em {OUTPUT_PATH}...")
    df_results.to_parquet(OUTPUT_PATH, index=False, compression='snappy')
    print(f"✓ {len(df_results):,} distâncias calculadas e salvas")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
