import pandas as pd
import json
from tqdm import tqdm

# CSV oku
csv_df = pd.read_csv("veri.csv")

# JSONL oku ve merge
merged = []
buffer = []
BUFFER_SIZE = 1000  # her 1000 satırda diske yaz

with open("veri.jsonl", "r", encoding="utf-8") as f, \
     open("birlesik.jsonl", "w", encoding="utf-8") as out_f:

    for i, line in enumerate(tqdm(f, desc="Merging JSONL + CSV")):
        item = json.loads(line)

        if i < len(csv_df):
            row_dict = csv_df.iloc[i].to_dict()
            item.update(row_dict)

        buffer.append(json.dumps(item, ensure_ascii=False))

        # buffer dolunca yaz
        if len(buffer) >= BUFFER_SIZE:
            out_f.write("\n".join(buffer) + "\n")
            buffer = []

    # kalanları yaz
    if buffer:
        out_f.write("\n".join(buffer) + "\n")
