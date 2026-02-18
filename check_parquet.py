import pandas as pd
import sys

try:
    df = pd.read_parquet(sys.argv[1])
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
except Exception as e:
    print(f"Error reading {sys.argv[1]}: {e}")
