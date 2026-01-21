import pandas as pd
from pathlib import Path

# paths
csvs = [
    Path("processed_output_data/Climate_Amazon_North_1961-1980.csv"),
    Path("processed_output_data/Climate_Amazon_North_1981-2000.csv"),
    Path("processed_output_data/Climate_Amazon_North_2001-2024.csv")
]

# load e concat
frames = [pd.read_csv(csv) for csv in csvs]
df = pd.concat(frames, ignore_index=True)

# order by city
if 'CD_MUN' in df.columns and 'year' in df.columns:
    df = df.sort_values(['CD_MUN', 'year'])

# save 
output = Path("processed_output_data/Climate_Amazon_North_1961-2024.csv")
df.to_csv(output, index=False, encoding='utf-8-sig')

print(f"Arquivo salvo: {output}")
