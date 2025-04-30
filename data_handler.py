import pandas as pd 
def load_data(filepath):
    encodings_to_try = ['utf-8', 'ISO-8859-1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Failed to decode {filepath}")

def save_data(data, filepath):
    return data.to_csv(filepath, index=False)