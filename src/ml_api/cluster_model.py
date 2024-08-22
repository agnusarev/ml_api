from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

DATA_NUMERIC = ["Age", "Credit amount", "Duration"]
FILLNA_COLS = ["Saving accounts", "Checking account"]
DROP_COLUMNS = ["Unnamed: 0"]


def read_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocessing_data(df: pd.DataFrame) -> pd.DataFrame:
    df[FILLNA_COLS] = df[FILLNA_COLS].fillna("unknown")
    df = df.drop(DROP_COLUMNS, axis=1)
    df[DATA_NUMERIC] = np.log(df[DATA_NUMERIC])
    for c in df.columns:
        if df[c].dtype == "object":
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df)
    return pd.DataFrame(X_scaled, columns=df.columns)


def linear_dimensionality_reduction(df: pd.DataFrame) -> np.ndarray:
    pca = PCA(n_components=2)
    return pca.fit_transform(df)


def clustering_data(df: pd.DataFrame) -> pd.DataFrame:
    _proc_data = preprocessing_data(df)
    _pca_data = linear_dimensionality_reduction(_proc_data)
    kmeans = KMeans(n_clusters=2, random_state=10).fit(_pca_data)
    df["Cluster"] = pd.Series(kmeans.labels_, dtype="int64")
    return df


if __name__ == "__main__":
    _path = Path(
        Path(__file__).resolve().parent.parent.parent
        / "tests/data/german_credit_data.csv"
    )
    _data = read_data(_path)
    clustering_data(_data).to_csv(
        Path(__file__).resolve().parent.parent.parent
        / "tests/data/german_credit_data_testing.csv",
        index=False,
    )
