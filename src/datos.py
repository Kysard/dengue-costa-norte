"""Preparación de datos: variables, partición temporal y ventanas deslizantes."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

OBJETIVOS = {"dengue": "dengue_total", "eda": "eda_men5"}
F_HISTORIA = ["y", "sin_sem", "cos_sem"]
F_CLIMA = ["t2m_media", "t2m_max", "t2m_min", "precip_log", "hr_media"]
F_ENSO = ["nino12_anom", "oni"]

# Partición temporal (por año de la semana objetivo)
ANIO_FIN_TRAIN = 2016
ANIO_FIN_VAL = 2020


def particion(anio):
    if anio <= ANIO_FIN_TRAIN:
        return "train"
    if anio <= ANIO_FIN_VAL:
        return "val"
    return "test"


def rezagar_enso(df: pd.DataFrame) -> pd.DataFrame:
    """Los índices de El Niño se publican con retraso. El Niño 1+2 de un mes sale a inicios del mes
    siguiente y el ONI (promedio móvil de 3 meses, asignado a su mes central) recién se completa un mes
    después de ese mes central. Para no usar información que todavía no existía al momento de pronosticar,
    cada semana usa el Niño 1+2 del mes anterior y el ONI centrado dos meses antes."""
    mitad = pd.to_datetime(df["fecha_inicio"]) + pd.Timedelta(days=3)
    mes = (mitad.dt.year * 12 + mitad.dt.month - 1).to_numpy()
    for col, rezago in (("nino12_anom", 1), ("oni", 2)):
        tabla = pd.Series(df[col].to_numpy(), index=mes).groupby(level=0).first()
        df[col] = pd.Series(mes - rezago).map(tabla).fillna(tabla.iloc[0]).to_numpy()
    return df


def preparar(panel: pd.DataFrame, objetivo: str) -> pd.DataFrame:
    df = panel.copy().sort_values(["ubigeo_prov", "ano", "semana"]).reset_index(drop=True)
    df = rezagar_enso(df)
    df["casos"] = df[OBJETIVOS[objetivo]].astype(float)
    df["y"] = np.log1p(df["casos"])
    df["precip_log"] = np.log1p(df["precip_mm"].clip(lower=0))
    ang = 2 * np.pi * (df["semana"] - 1) / 52.18
    df["sin_sem"], df["cos_sem"] = np.sin(ang), np.cos(ang)
    provs = sorted(df["ubigeo_prov"].unique())
    df["prov_idx"] = df["ubigeo_prov"].map({p: i for i, p in enumerate(provs)})
    df["split"] = df["ano"].map(particion)
    return df


def lista_variables(con_clima=True, con_enso=True):
    f = list(F_HISTORIA)
    if con_clima:
        f += F_CLIMA
    if con_enso:
        f += F_ENSO
    return f


@dataclass
class Escalador:
    media: np.ndarray
    desv: np.ndarray

    @classmethod
    def ajustar(cls, df, cols):
        tr = df[df["split"] == "train"][cols].to_numpy(dtype=float)
        return cls(tr.mean(0), tr.std(0) + 1e-6)

    def transformar(self, a):
        return (a - self.media) / self.desv


@dataclass
class Ventanas:
    X: np.ndarray          # (N, L, F)  variables estandarizadas
    P: np.ndarray          # (N,)       índice de provincia
    Y: np.ndarray          # (N, H)     log1p(casos) futuros (sin estandarizar)
    origen: np.ndarray     # (N,)       fila del panel de la última semana observada
    split: np.ndarray      # (N,)       train / val / test
    ult: np.ndarray        # (N,)       log1p(casos) de la última semana observada


def crear_ventanas(df: pd.DataFrame, variables, L=26, H=4, escalador: Escalador = None):
    """Para cada provincia y semana t: entrada = semanas t-L+1..t, salida = semanas t+1..t+H.
    Una ventana pertenece a un conjunto solo si sus H semanas objetivo caen en ese conjunto
    (las ventanas que cruzan el límite entre conjuntos se descartan)."""
    if escalador is None:
        escalador = Escalador.ajustar(df, variables)
    Xs, Ps, Ys, Os, Ss, Us = [], [], [], [], [], []
    for _, g in df.groupby("prov_idx", sort=True):
        x = escalador.transformar(g[variables].to_numpy(dtype=float)).astype(np.float32)
        y = g["y"].to_numpy(dtype=np.float32)
        sp = g["split"].to_numpy()
        idx = g.index.to_numpy()
        p = int(g["prov_idx"].iloc[0])
        for t in range(L - 1, len(g) - H):
            s_obj = sp[t + 1:t + 1 + H]
            if not (s_obj == s_obj[0]).all():
                continue
            Xs.append(x[t - L + 1:t + 1])
            Ys.append(y[t + 1:t + 1 + H])
            Ps.append(p)
            Os.append(idx[t])
            Ss.append(s_obj[0])
            Us.append(y[t])
    v = Ventanas(np.stack(Xs), np.array(Ps), np.stack(Ys), np.array(Os), np.array(Ss),
                 np.array(Us, dtype=np.float32))
    return v, escalador


def subconjunto(v: Ventanas, nombre: str) -> Ventanas:
    m = v.split == nombre
    return Ventanas(v.X[m], v.P[m], v.Y[m], v.origen[m], v.split[m], v.ult[m])
