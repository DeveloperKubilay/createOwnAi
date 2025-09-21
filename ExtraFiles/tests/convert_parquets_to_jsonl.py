import os
import json
import pandas as pd
import requests
import gzip
import io
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

input_dir = "./javascript_parquets/data/JavaScript"
output_path = "/mnt/500gb/all_JavaScript.jsonl"

parquet_files = sorted([
    os.path.join(input_dir, f) 
    for f in os.listdir(input_dir) 
    if f.endswith(".parquet")
])

def fetch_s3_content(blob_id):
    url = f"https://softwareheritage.s3.amazonaws.com/content/{blob_id}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.ok:
            with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as f:
                return blob_id, f.read().decode("utf-8")
        else:
            return blob_id, None
    except Exception as e:
        return blob_id, None

with open(output_path, "w", encoding="utf-8") as out_file:
    for file in tqdm(parquet_files, desc="Veriler dönüştürülüyor"):
        try:
            df = pd.read_parquet(file)
            blob_ids = df["blob_id"].tolist()
            # 8 THREAD ÇALIŞTIR (isteğe göre artır)
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_blob = {executor.submit(fetch_s3_content, blob_id): blob_id for blob_id in blob_ids}
                for future in tqdm(as_completed(future_to_blob), total=len(blob_ids), desc="S3'ten paralel çekiliyor", leave=False):
                    blob_id, content = future.result()
                    if content:
                        clean = content.strip()
                        if clean:
                            out_file.write(json.dumps({"content": clean}) + "\n")
        except Exception as e:
            print(f"Hata: {file} -> {e}")