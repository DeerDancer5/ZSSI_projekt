import pandas as pd


def extract_title(name: str) -> str:
    if pd.isna(name):
        return "Unknown"
    return name.split(",")[1].split(".")[0].strip()

def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Title"] = df["Name"].apply(extract_title)

    return df