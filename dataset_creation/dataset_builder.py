"""
dataset_builder.py
------------------
Converts intent-labeled CSV into JSONL format for LLM training.
"""

import json
import pandas as pd
from config import INTENT_DATA_PATH, STRUCTURED_DATA_PATH

def build_dialogue_dataset():
    print("🧱 Building structured dialogue dataset...")
    df = pd.read_csv(INTENT_DATA_PATH)
    df = df.dropna(subset=["answer"])

    records = []
    for i, row in df.iterrows():
        records.append({
            "id": i + 1,
            "intent": row.get("intent", "general_query"),
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]}
            ]
        })

    STRUCTURED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STRUCTURED_DATA_PATH, "w", encoding="utf-8") as f:
        for item in records:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Structured dataset created → {STRUCTURED_DATA_PATH}")
    print(f"📊 Total dialogue pairs: {len(records)}")

if __name__ == "__main__":
    build_dialogue_dataset()
