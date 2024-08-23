from typing import List
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from ml_api.cluster_model import clustering_data
from pandas_to_pydantic import dataframe_to_pydantic

app = FastAPI()


class Item(BaseModel):
    id: int
    age: int
    sex: str
    job: int
    housing: str
    save_account: str
    check_account: str
    credit_amount: int
    duration: int
    purpose: str
    cluster: int | None = None


@app.post("/cluster/")
def cluster_data(items: List[Item]) -> None:
    df = pd.DataFrame([item.model_dump() for item in items])
    _cluster_df = clustering_data(df)
    return dataframe_to_pydantic(_cluster_df, Item)
