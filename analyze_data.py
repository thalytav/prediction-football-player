import pandas as pd
import json
import numpy as np

# Load data
df = pd.read_csv('cleaned_football_data.csv')
print(f'Total rows: {len(df)}')
print(f'\nColumns: {list(df.columns)}')

# Parse embeddings
df['embedding'] = df['embedding'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

# Position distribution
df['primary_position'] = df['positions'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else x)
print(f'\nPosition distribution:')
print(df['primary_position'].value_counts())

# Embedding dimensions
emb_lens = df['embedding'].apply(len)
print(f'\nEmbedding dimensions: min={emb_lens.min()}, max={emb_lens.max()}, mean={emb_lens.mean():.1f}')

# Missing values
print(f'\nMissing values:')
print(df.isnull().sum()[df.isnull().sum() > 0])

# Check numeric stats columns
stats_cols = ['age', 'acceleration', 'sprint_speed', 'dribbling', 
              'short_passing', 'finishing', 'stamina', 'strength']
available_stats = [col for col in stats_cols if col in df.columns]
print(f'\nAvailable stat columns: {available_stats}')

if available_stats:
    print(f'\nStats summary:')
    print(df[available_stats].describe())
