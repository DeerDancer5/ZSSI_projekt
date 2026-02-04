import pandas as pd


def load_titanic(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def save_dataframe(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)