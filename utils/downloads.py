import os
import requests

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading {dest_path}...")
        r = requests.get(url)
        with open(dest_path, "wb") as f:
            f.write(r.content)
        print(f"Downloaded to {dest_path}")
    else:
        print(f"{dest_path} already exists.")
