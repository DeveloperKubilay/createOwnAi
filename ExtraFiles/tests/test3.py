import requests
import gzip
import io

blob_id = "e7b20b09e4f253de74034bcbf9f3c31a0258f5c3"
url = f"https://softwareheritage.s3.amazonaws.com/content/{blob_id}"

# Dosyayı indir
resp = requests.get(url)
if resp.ok:
    # Gzip'i aç
    with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as f:
        content = f.read().decode("utf-8")
        print(content[:500])  # İlk 500 karakteri göster, istersen tamamını yazdır
else:
    print("Dosya çekilemedi 😭")