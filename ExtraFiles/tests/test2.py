import os
from dotenv import load_dotenv
import boto3
from smart_open import open
from datasets import load_dataset

# .env dosyasını yükle
load_dotenv()

session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
s3 = session.client("s3")

def download_contents(blob_id, src_encoding):
    s3_url = f"s3://softwareheritage/content/{blob_id}"
    
    with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
        content = fin.read().decode(src_encoding)
    
    return {"content": content}

ds = load_dataset("bigcode/the-stack-v2", split="train", streaming=True)
ds = ds.map(lambda row: download_contents(row["blob_id"], row["src_encoding"]))
for row in ds:
    print(row["content"])
    break
