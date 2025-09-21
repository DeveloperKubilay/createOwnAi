from huggingface_hub import hf_hub_download, list_repo_files
from dotenv import load_dotenv
import os

# .env dosyasını yükle
load_dotenv()

repo_id = "bigcode/the-stack-v2"
subfolder = "data/JavaScript"
local_dir = "../javascript_parquets"
token = os.getenv("HF_TOKEN")

os.makedirs(local_dir, exist_ok=True)

# Tüm dosyaları al
files = list_repo_files(repo_id, repo_type="dataset", token=token)

# Sadece ilgili klasördeki .parquet dosyaları filtrele
javascript_files = [f for f in files if f.startswith(subfolder) and f.endswith(".parquet")]

print(f"{len(javascript_files)} tane JavaScript parquet dosyası bulundu.")

# Bulduğu tüm dosyaları indir
for file in javascript_files:
    print(f"Downloading {file}...")
    hf_hub_download(
        repo_id=repo_id,
        filename=file,
        local_dir=local_dir,
        repo_type="dataset",
        token=token
    )
