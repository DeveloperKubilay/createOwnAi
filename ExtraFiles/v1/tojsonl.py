import os
import json
import sys
import requests
import gzip
import io
import time
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import pyarrow.parquet as pq
import gc

# Ayarlar
input_dir = "/mnt/500gb/javascript_parquets/data/JavaScript"
output_path = "/mnt/500gb/all_JavaScript.jsonl"
max_threads = 32
max_inflight = 256  # aynı anda en fazla bu kadar future bellekte olsun
max_retries = 5
batch_size = 1000  # daha sık flush, RAM baskısını azaltır

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
            buffer = []
            # Parquet'tan yalnızca blob_id oku, pyarrow ile row-group bazlı oku (daha az RAM)
            pf = pq.ParquetFile(file)
            row_groups = range(pf.num_row_groups)

            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                for rg in tqdm(row_groups, desc="Parquet row-groupları işleniyor"):
                    table = pf.read_row_group(rg, columns=['blob_id'])
                    blob_ids = table.column('blob_id').to_pylist()

                    pending = set()
                    for blob_id in tqdm(blob_ids, desc="S3'ten paralel çekiliyor", leave=False):
                        fut = executor.submit(fetch_s3_content, blob_id)
                        pending.add(fut)

                        if len(pending) >= max_inflight:
                            done, pending = wait(pending, return_when=FIRST_COMPLETED)
                            for d in done:
                                b_id, content = d.result()
                                total_blobs_processed += 1
                                if content:
                                    clean = content.strip()
                                    if clean:
                                        buffer.append(json.dumps({"content": clean}))

                                if len(buffer) >= batch_size:
                                    out_file.write("\n".join(buffer) + "\n")
                                    buffer.clear()

                                if total_blobs_processed % 10000 == 0:
                                    print(f"{total_blobs_processed} blob işlendi")

                    # Kalan pending'leri bekle
                    if pending:
                        done, _ = wait(pending)
                        for d in done:
                            b_id, content = d.result()
                            total_blobs_processed += 1
                            if content:
                                clean = content.strip()
                                if clean:
                                    buffer.append(json.dumps({"content": clean}))

                            if len(buffer) >= batch_size:
                                out_file.write("\n".join(buffer) + "\n")
                                buffer.clear()

                            if total_blobs_processed % 10000 == 0:
                                print(f"{total_blobs_processed} blob işlendi")

                    # row-group/dosya içi değişkenleri sil ve GC çağır
                    try:
                        del blob_ids
                    except Exception:
                        pass
                    try:
                        del table
                    except Exception:
                        pass
                    gc.collect()

                    # Kalan buffer'ı yaz
                    if buffer:
                        out_file.write("\n".join(buffer) + "\n")
                        buffer.clear()

                # dosya düzeyinde bellek temizliği
                try:
                    del buffer
                except Exception:
                    pass
                gc.collect()

        except Exception as e:
            print(f"Hata: {file} -> {e}")