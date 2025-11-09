# dataset_loader.py
import pandas as pd
from datasets import Dataset
from config import DATASET_PATH

def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    # Combine question + body as input text, answer as target
    df["input_text"] = df["question"].fillna("") + " " + df["body"].fillna("")
    df["target_text"] = df["answer"].fillna("")
    df = df[["input_text", "target_text"]]
    return Dataset.from_pandas(df)
