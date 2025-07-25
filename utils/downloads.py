import os
import requests

MODEL_URLS = {
    "model": "https://1drv.ms/u/c/6c724deadf3112a0/Edpn_ZWxDi1Kl3C9617z2N4BoiFwp4cM4PzUSq-Foci8Ww?e=bni1YI",
    "scaler": "https://1drv.ms/u/c/6c724deadf3112a0/EVdJ5eC0fdZAvcAcltwyLq0Bmo3ar50eB3b5L_aJzaBn6w?e=2R0kWG",
    "encoders": "https://1drv.ms/u/c/6c724deadf3112a0/EWuMBX-bAmpMsM2qrf21W2kBrReBpEyk3-MaDoDyB0JduQ?e=S9Rrsu"
}

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading {dest_path}...")
        with requests.get(url) as r:
            if r.status_code == 200:
                with open(dest_path, 'wb') as f:
                    f.write(r.content)
            else:
                raise Exception(f"Failed to download {url}, status code {r.status_code}")

def maybe_download_model():
    os.makedirs("models", exist_ok=True)
    download_file(MODEL_URLS["model"], "models/price_model.pkl")
    download_file(MODEL_URLS["scaler"], "models/scaler.pkl")
    download_file(MODEL_URLS["encoders"], "models/encoders.pkl")
