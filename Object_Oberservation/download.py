import os
import requests
from tqdm import tqdm  # Für Fortschrittsbalken

MODEL_URLS = {
    "depth_model.pt": "https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-f6b98070.pt",
}

def download_model(url, filename):
    # Prüfe, ob die Datei bereits existiert
    if os.path.exists(filename):
        print(f"{filename} existiert bereits – Überspringe Download.")
        return

    # Download mit Fortschrittsbalken
    print(f"Lade {filename} herunter...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
            bar.update(len(data))

if __name__ == "__main__":
    for filename, url in MODEL_URLS.items():
        download_model(url, filename)
    print("Alle Downloads abgeschlossen!")