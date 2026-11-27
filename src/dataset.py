from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .audio import process_audio

def load_metadata(metadata_file):
    csv_path = Path(metadata_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file {csv_path} không tồn tại.")

    df = pd.read_csv(csv_path)

    required_columns = {"filename", "fold", "target", "category"}

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Thiếu các cột trong tệp metadata: {missing_columns}")

    if df.isnull().any().any():
        raise ValueError("Tệp metadata chứa giá trị null.")

    if df.duplicated().any():
        raise ValueError("Tệp metadata chứa các hàng trùng lặp.")

    return df

def split_by_fold(df, val_fold, test_fold):
    if val_fold == test_fold:
        raise ValueError("val_fold và test_fold không thể giống nhau.")

    valid_folds = {1, 2, 3, 4, 5}

    if val_fold not in valid_folds or test_fold not in valid_folds:
        raise ValueError(f"val_fold và test_fold phải nằm trong {valid_folds}.")

    train_df = df[~df["fold"].isin([val_fold, test_fold])].copy()
    val_df = df[df["fold"] == val_fold].copy()
    test_df = df[df["fold"] == test_fold].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Một trong các tập dữ liệu train, val hoặc test trống.")

    return train_df, val_df, test_df
