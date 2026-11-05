"""
Construye el panel provincia-semana (costa norte del Perú, 2000-2025).

Entradas (carpeta data/):
  raw/dengue_provincia_semana_peru_2000_2025.csv      casos de dengue agregados (CDC Perú)
  raw/eda_provincia_semana_costanorte_2000_2025.csv   episodios de EDA agregados (CDC Perú)
  raw/clima_nasa_power_diario_costanorte_1999_2025.csv clima diario (NASA POWER)
  raw/oni.ascii.txt                                   índice ONI (NOAA CPC)
  raw/sstoi.indices  o  raw/nino12_cpc_1999_2026.txt  Niño 1+2 (NOAA CPC)
  external/provincias.csv, external/poblacion_censos.csv

Salida:
  processed/panel_provincia_semana.csv

Uso:
  python src/construir_dataset.py --data data
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

GRAVEDAD = {"DENGUE SIN SIGNOS DE ALARMA": "SSA",
            "DENGUE CON SIGNOS DE ALARMA": "CSA",
            "DENGUE GRAVE": "GRAVE"}


# ---------------------------------------------------------------------------
# Calendario epidemiológico (semana domingo-sábado; la semana 1 es la primera
# que tiene al menos 4 días del año nuevo, igual que el calendario del CDC Perú)
# ---------------------------------------------------------------------------
def inicio_se1(anio: int) -> date:
    ene1 = date(anio, 1, 1)
    dow = (ene1.weekday() + 1) % 7          # domingo = 0
    return ene1 - timedelta(days=dow) if dow <= 3 else ene1 + timedelta(days=7 - dow)


def semanas_del_anio(anio: int) -> int:
    return (inicio_se1(anio + 1) - inicio_se1(anio)).days // 7


def semana_epi(d: date):
    anio = d.year
    if d >= inicio_se1(anio + 1):
        anio += 1
    elif d < inicio_se1(anio):
        anio -= 1
    return anio, (d - inicio_se1(anio)).days // 7 + 1


def calendario(anio_ini=2000, anio_fin=2025) -> pd.DataFrame:
    filas = []
    for a in range(anio_ini, anio_fin + 1):
        for s in range(1, semanas_del_anio(a) + 1):
            ini = inicio_se1(a) + timedelta(weeks=s - 1)
            filas.append((a, s, pd.Timestamp(ini)))
    return pd.DataFrame(filas, columns=["ano", "semana", "fecha_inicio"])


def corregir_semana_53(df: pd.DataFrame) -> pd.DataFrame:
    """Algunos registros traen semana 53 en años que solo tienen 52; se pasan a la 52."""
    df = df.copy()
    max_sem = df["ano"].map(semanas_del_anio)
    df.loc[df["semana"] > max_sem, "semana"] = max_sem
    return df


def corregir_texto(s: pd.Series) -> pd.Series:
    # La fuente trae la Ñ dañada (carácter U+FFFD). Solo afecta a la Ñ.
    return s.str.replace("�", "Ñ", regex=False)


# ---------------------------------------------------------------------------
# Agregación desde los CSV originales del CDC (se usa en descargar_datos.py)
# ---------------------------------------------------------------------------
def agregar_dengue_original(ruta_csv) -> pd.DataFrame:
    d = pd.read_csv(ruta_csv, dtype=str, encoding="utf-8", encoding_errors="replace",
                    on_bad_lines="skip", engine="python")
    d["ubigeo_prov"] = d["ubigeo"].str.zfill(6).str[:4]
    d["gravedad"] = d["enfermedad"].map(GRAVEDAD)
    d["ano"] = d["ano"].astype(int)
    d["semana"] = d["semana"].astype(int)
    g = (d.groupby(["departamento", "provincia", "ubigeo_prov", "ano", "semana", "gravedad", "tipo_dx"])
           .size().rename("casos").reset_index())
    return g


def agregar_eda_original(ruta_csv, prefijos=("24", "20", "14")) -> pd.DataFrame:
    d = pd.read_csv(ruta_csv, dtype=str, encoding="utf-8", encoding_errors="replace",
                    on_bad_lines="skip", engine="python")
    d["ubigeo_prov"] = d["ubigeo"].str.zfill(6).str[:4]
    d = d[d["ubigeo_prov"].str[:2].isin(prefijos)]
    num = ["episodios_men5", "hospitalizados_men5", "defunciones_men5",
           "episodios_may5", "hospitalizados_may5", "defunciones_may5"]
    d[num] = d[num].apply(pd.to_numeric, errors="coerce").fillna(0)
    d["ano"] = d["ano"].astype(int)
    d["semana"] = d["semana"].astype(int)
    return d.groupby(["departamento", "provincia", "ubigeo_prov", "ano", "semana"])[num].sum().reset_index()


# ---------------------------------------------------------------------------
# Lectura de índices El Niño
# ---------------------------------------------------------------------------
def leer_oni(ruta) -> pd.DataFrame:
    t = pd.read_csv(ruta, sep=r"\s+")
    estaciones = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
    t["mes"] = t["SEAS"].map({e: i + 1 for i, e in enumerate(estaciones)})  # mes central
    return t.rename(columns={"YR": "ano_m", "ANOM": "oni"})[["ano_m", "mes", "oni"]]


def leer_nino12(data: Path) -> pd.DataFrame:
    original = data / "raw" / "sstoi.indices"
    if original.exists():
        t = pd.read_csv(original, sep=r"\s+")
        t.columns = ["ano_m", "mes", "nino12_sst", "nino12_anom", "n3", "n3a", "n4", "n4a", "n34", "n34a"]
        return t[["ano_m", "mes", "nino12_sst", "nino12_anom"]]
    txt = (data / "raw" / "nino12_cpc_1999_2026.txt").read_text().strip()
    filas = [r.split() for r in txt.split(";")]
    t = pd.DataFrame(filas, columns=["yyyymm", "nino12_sst", "nino12_anom"])
    t["ano_m"] = t["yyyymm"].str[:4].astype(int)
    t["mes"] = t["yyyymm"].str[4:].astype(int)
    t[["nino12_sst", "nino12_anom"]] = t[["nino12_sst", "nino12_anom"]].astype(float)
    return t[["ano_m", "mes", "nino12_sst", "nino12_anom"]]


# ---------------------------------------------------------------------------
# Población anual: interpolación geométrica entre censos (1993, 2007, 2017)
# y extrapolación con la tasa 2007-2017 después de 2017
# ---------------------------------------------------------------------------
def poblacion_anual(pob: pd.DataFrame, anios) -> pd.DataFrame:
    filas = []
    for _, r in pob.iterrows():
        for a in anios:
            if a < 2007:
                p = r.censo_1993 * (r.censo_2007 / r.censo_1993) ** ((a - 1993) / 14)
            else:
                p = r.censo_2007 * (r.censo_2017 / r.censo_2007) ** ((a - 2007) / 10)
            filas.append((str(r.ubigeo_prov), a, round(p)))
    return pd.DataFrame(filas, columns=["ubigeo_prov", "ano", "poblacion"])


# ---------------------------------------------------------------------------
def construir(data: Path) -> pd.DataFrame:
    prov = pd.read_csv(data / "external" / "provincias.csv", dtype={"ubigeo_prov": str})
    cod = prov["ubigeo_prov"].tolist()

    # 1) Base: 14 provincias x todas las semanas epidemiológicas 2000-2025
    cal = calendario(2000, 2025)
    panel = prov[["ubigeo_prov", "departamento", "provincia"]].merge(cal, how="cross")

    # 2) Dengue
    den = pd.read_csv(data / "raw" / "dengue_provincia_semana_peru_2000_2025.csv", dtype={"ubigeo_prov": str})
    den = corregir_semana_53(den[den["ubigeo_prov"].isin(cod)])
    piv = den.pivot_table(index=["ubigeo_prov", "ano", "semana"], columns="gravedad",
                          values="casos", aggfunc="sum", fill_value=0)
    piv = piv.rename(columns={"SSA": "dengue_ssa", "CSA": "dengue_csa", "GRAVE": "dengue_grave"})
    tdx = den.pivot_table(index=["ubigeo_prov", "ano", "semana"], columns="tipo_dx",
                          values="casos", aggfunc="sum", fill_value=0)
    tdx = tdx.rename(columns={"C": "dengue_confirmados", "P": "dengue_probables"})
    panel = panel.merge(piv.reset_index(), on=["ubigeo_prov", "ano", "semana"], how="left")
    panel = panel.merge(tdx.reset_index(), on=["ubigeo_prov", "ano", "semana"], how="left")

    # 3) EDA
    eda = pd.read_csv(data / "raw" / "eda_provincia_semana_costanorte_2000_2025.csv", dtype={"ubigeo_prov": str})
    eda = corregir_semana_53(eda[eda["ubigeo_prov"].isin(cod)])
    eda = eda.groupby(["ubigeo_prov", "ano", "semana"]).sum(numeric_only=True).reset_index()
    eda["eda_defunciones"] = eda["defunciones_men5"] + eda["defunciones_may5"]
    eda = eda.rename(columns={"episodios_men5": "eda_men5", "episodios_may5": "eda_may5",
                              "hospitalizados_men5": "eda_hosp_men5", "hospitalizados_may5": "eda_hosp_may5"})
    panel = panel.merge(eda[["ubigeo_prov", "ano", "semana", "eda_men5", "eda_may5",
                             "eda_hosp_men5", "eda_hosp_may5", "eda_defunciones"]],
                        on=["ubigeo_prov", "ano", "semana"], how="left")

    conteos = ["dengue_ssa", "dengue_csa", "dengue_grave", "dengue_confirmados", "dengue_probables",
               "eda_men5", "eda_may5", "eda_hosp_men5", "eda_hosp_may5", "eda_defunciones"]
    for c in conteos:
        if c not in panel:
            panel[c] = 0
        panel[c] = panel[c].fillna(0).astype(int)
    panel["dengue_total"] = panel["dengue_ssa"] + panel["dengue_csa"] + panel["dengue_grave"]
    panel["eda_total"] = panel["eda_men5"] + panel["eda_may5"]

    # 4) Clima diario -> semana epidemiológica
    cli = pd.read_csv(data / "raw" / "clima_nasa_power_diario_costanorte_1999_2025.csv", dtype={"ubigeo_prov": str})
    cli["fecha"] = pd.to_datetime(cli["fecha"].astype(str), format="%Y%m%d")
    se = cli["fecha"].dt.date.map(semana_epi)
    cli["ano"] = [x[0] for x in se]
    cli["semana"] = [x[1] for x in se]
    cli = cli.replace(-999, np.nan)
    sem = (cli.groupby(["ubigeo_prov", "ano", "semana"])
              .agg(t2m_media=("t2m", "mean"), t2m_max=("t2m_max", "mean"), t2m_min=("t2m_min", "mean"),
                   precip_mm=("prectotcorr", "sum"), hr_media=("rh2m", "mean"), n_dias=("t2m", "count"))
              .reset_index())
    # la última semana de 2025 termina en 2026 y solo tiene 4 días con datos: se lleva la lluvia a 7 días
    sem["precip_mm"] = sem["precip_mm"] * 7 / sem["n_dias"]
    sem = sem.drop(columns="n_dias").round(2)
    panel = panel.merge(sem, on=["ubigeo_prov", "ano", "semana"], how="left")

    # 5) El Niño (mes del miércoles de la semana)
    mitad = panel["fecha_inicio"] + pd.Timedelta(days=3)
    panel["ano_m"], panel["mes"] = mitad.dt.year, mitad.dt.month
    panel = panel.merge(leer_nino12(data), on=["ano_m", "mes"], how="left")
    panel = panel.merge(leer_oni(data / "raw" / "oni.ascii.txt"), on=["ano_m", "mes"], how="left")

    # 6) Población
    pob = pd.read_csv(data / "external" / "poblacion_censos.csv", dtype={"ubigeo_prov": str})
    panel = panel.merge(poblacion_anual(pob, range(2000, 2026)), on=["ubigeo_prov", "ano"], how="left")
    panel["inc_dengue_100k"] = (panel["dengue_total"] / panel["poblacion"] * 1e5).round(3)
    panel["inc_eda_100k"] = (panel["eda_total"] / panel["poblacion"] * 1e5).round(3)

    panel["provincia"] = corregir_texto(panel["provincia"])
    columnas = ["ubigeo_prov", "departamento", "provincia", "ano", "semana", "fecha_inicio",
                "dengue_ssa", "dengue_csa", "dengue_grave", "dengue_total",
                "dengue_confirmados", "dengue_probables",
                "eda_men5", "eda_may5", "eda_total", "eda_hosp_men5", "eda_hosp_may5", "eda_defunciones",
                "t2m_media", "t2m_max", "t2m_min", "precip_mm", "hr_media",
                "nino12_anom", "nino12_sst", "oni",
                "poblacion", "inc_dengue_100k", "inc_eda_100k"]
    panel = panel[columnas].sort_values(["ubigeo_prov", "ano", "semana"]).reset_index(drop=True)
    panel["fecha_inicio"] = panel["fecha_inicio"].dt.date
    return panel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    data = Path(args.data)
    panel = construir(data)
    (data / "processed").mkdir(parents=True, exist_ok=True)
    salida = data / "processed" / "panel_provincia_semana.csv"
    panel.to_csv(salida, index=False, encoding="utf-8")
    print(f"Panel guardado en {salida}: {panel.shape[0]} filas x {panel.shape[1]} columnas")
    print(f"Casos de dengue: {panel.dengue_total.sum():,} | episodios de EDA: {panel.eda_total.sum():,}")
    print(f"Valores faltantes por columna:\n{panel.isna().sum()[panel.isna().sum() > 0]}")
