import os
import json
import pandas as pd
import requests
import gzip
import io
import time
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ayarlar
input_dir = "/mnt/500gb/javascript_parquets/data/JavaScript"
output_path = "/mnt/500gb/all_JavaScript.jsonl"
max_threads = 32  # 100 thread ayarlandı
max_retries = 5
batch_size = 1000  # her 1000 satırda bir file flush

# Tüm parquet dosyalarını sırala
parquet_files = sorted([
    os.path.join(input_dir, f)
    for f in os.listdir(input_dir)
    if f.endswith(".parquet")
])

def fetch_s3_content(blob_id):
    url = f"https://softwareheritage.s3.amazonaws.com/content/{blob_id}"
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=10)
            if resp.ok:
                with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as f:
                    # UTF-8 hatalarını ignore ederek decode
                    return blob_id, f.read().decode("utf-8", errors="ignore")
            else:
                print(f"Uyarı: {blob_id} -> HTTP {resp.status_code}, retry {attempt+1}")
        except Exception as e:
            print(f"Hata: {blob_id} -> {e}, retry {attempt+1}")
        # Exponential backoff + random jitter
        time.sleep(2 ** attempt + random.random())
    print(f"Başarısız: {blob_id} tüm {max_retries} denemede")
    return blob_id, None

# Ana işleme
with open(output_path, "w", encoding="utf-8") as out_file:
    total_blobs_processed = 0
    for file in tqdm(parquet_files, desc="Veriler dönüştürülüyor"):
        try:
            df = pd.read_parquet(file)
            blob_ids = df["blob_id"].tolist()
            buffer = []

            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                future_to_blob = {executor.submit(fetch_s3_content, blob_id): blob_id for blob_id in blob_ids}
                
                for future in tqdm(as_completed(future_to_blob), total=len(blob_ids), desc="S3'ten paralel çekiliyor", leave=False):
                    blob_id, content = future.result()
                    total_blobs_processed += 1

                    if content:
                        clean = content.strip()
                        if clean:
                            buffer.append(json.dumps({"content": clean}))

                    # Belli sayıda satır biriktiğinde dosyaya yaz
                    if len(buffer) >= batch_size:
                        out_file.write("\n".join(buffer) + "\n")
                        buffer.clear()

                    # Log progress her 10k blob'da
                    if total_blobs_processed % 10000 == 0:
                        print(f"{total_blobs_processed} blob işlendi")

            # Kalan buffer'ı yaz
            if buffer:
                out_file.write("\n".join(buffer) + "\n")
                buffer.clear()

        except Exception as e:
            print(f"Hata: {file} -> {e}")