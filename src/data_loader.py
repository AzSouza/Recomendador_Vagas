import json
import pandas as pd

def load_json(path: str) -> pd.DataFrame:
    # carrega o JSON
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # se vier um dicionário mapeando IDs → registros, transforme em lista
    if isinstance(data, dict):
        # se for algo como {"123": {...}, "456": {...}, ...}
        data = list(data.values())

    # normaliza a lista de dicts em DataFrame
    return pd.json_normalize(data)

def load_all(data_dir: str) -> dict:
    return {
        'vagas': load_json(f'{data_dir}/vagas.json'),
        'prospects': load_json(f'{data_dir}/prospects.json'),
        'applicants': load_json(f'{data_dir}/applicants.json'),
    }

if __name__ == '__main__':
    dfs = load_all('data')   # assumindo que você está no root do projeto
    for name, df in dfs.items():
        print(f"{name}: {df.shape}")
