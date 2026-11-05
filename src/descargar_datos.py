"""
Descarga los datos originales y genera los archivos de data/raw/ (el panel se arma después con
construir_dataset.py). Pensado para correr en Colab o en una PC con internet.

Uso:
  python src/descargar_datos.py --data data
  python src/construir_dataset.py --data data

Fuentes:
  CDC Perú (Plataforma Nacional de Datos Abiertos, licencia ODC-By)
  NASA POWER Daily API (uso libre citando al proyecto)
  NOAA Climate Prediction Center (dominio público)
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests

import construir_dataset as cd

URL_DENGUE = "https://www.datosabiertos.gob.pe/sites/default/files/datos_abiertos_vigilancia_dengue_2000_2025_0.csv"
URL_EDA = "https://www.datosabiertos.gob.pe/sites/default/files/datos_abiertos_vigilancia_edas_2000_2025.csv"
URL_ONI = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
URL_SST = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
URL_POWER = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS_POWER = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M"


def bajar(url, destino):
    destino = Path(destino)
    if destino.exists():
        print(f"ya existe {destino.name}")
        return destino
    print(f"descargando {url}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(destino, "wb") as f:
            for parte in r.iter_content(1 << 20):
                f.write(parte)
    return destino


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for parte in iter(lambda: f.read(1 << 20), b""):
            h.update(parte)
    return h.hexdigest()


def clima_nasa(provincias, inicio="19990101", fin="20251231"):
    filas = []
    for r in provincias.itertuples(index=False):
        params = {"parameters": PARAMS_POWER, "community": "AG", "longitude": r.lon, "latitude": r.lat,
                  "start": inicio, "end": fin, "format": "JSON"}
        for intento in range(3):
            try:
                j = requests.get(URL_POWER, params=params, timeout=300).json()
                break
            except Exception:
                time.sleep(5)
        p = j["properties"]["parameter"]
        lon, lat = j["geometry"]["coordinates"][:2]
        for dia in p["T2M"]:
            filas.append([r.ubigeo_prov, r.provincia, r.lat, r.lon, int(dia)] +
                         [p[k][dia] for k in PARAMS_POWER.split(",")] + [lat, lon])
        print(f"clima {r.provincia}: {len(p['T2M'])} días")
    cols = ["ubigeo_prov", "provincia", "lat", "lon", "fecha"] + PARAMS_POWER.lower().split(",") + ["grid_lat", "grid_lon"]
    return pd.DataFrame(filas, columns=cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    raw = Path(args.data) / "raw"
    orig = Path(args.data) / "original"
    raw.mkdir(parents=True, exist_ok=True)
    orig.mkdir(parents=True, exist_ok=True)

    f_den = bajar(URL_DENGUE, orig / "datos_abiertos_vigilancia_dengue_2000_2025.csv")
    f_eda = bajar(URL_EDA, orig / "datos_abiertos_vigilancia_edas_2000_2025.csv")
    bajar(URL_ONI, raw / "oni.ascii.txt")
    bajar(URL_SST, raw / "sstoi.indices")

    print("agregando dengue por provincia y semana...")
    cd.agregar_dengue_original(f_den).to_csv(raw / "dengue_provincia_semana_peru_2000_2025.csv", index=False)
    print("agregando EDA por provincia y semana...")
    cd.agregar_eda_original(f_eda).to_csv(raw / "eda_provincia_semana_costanorte_2000_2025.csv", index=False)

    prov = pd.read_csv(Path(args.data) / "external" / "provincias.csv", dtype={"ubigeo_prov": str})
    clima_nasa(prov).to_csv(raw / "clima_nasa_power_diario_costanorte_1999_2025.csv", index=False)

    json.dump({"dengue": {"url": URL_DENGUE, "sha256": sha256(f_den)},
               "eda": {"url": URL_EDA, "sha256": sha256(f_eda)}},
              open(raw / "manifest_descarga.json", "w"), indent=2)
    print("Listo. Ahora: python src/construir_dataset.py --data", args.data)


if __name__ == "__main__":
    main()
