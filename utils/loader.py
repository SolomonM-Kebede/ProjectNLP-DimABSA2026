import pandas as pd
import json
from pathlib import Path

def load_metrics(file_path):
    """
    Load metrics file for any subtask:
    - CSV for subtask1
    - JSON for subtask2/subtask3
    Returns:
        DataFrame for CSV
        dict for JSON
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
        return df
    elif file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def list_models(dataset_dir, subtask):
    """
    List available models for a given dataset/subtask
    Returns a list of filenames without extensions
    """
    path = Path(dataset_dir) / subtask
    if not path.exists():
        return []
    
    # Filter for .csv and .json only
    files = [f for f in path.iterdir() if f.suffix.lower() in [".csv", ".json"]]
    models = [f.stem for f in files]
    return models