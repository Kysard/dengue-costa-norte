"""Inferencia: pronóstico de las próximas H semanas desde la última semana disponible y
conversión a medicamentos con el kit por caso."""
import numpy as np
import pandas as pd
import torch

import datos as D
from entrenar import DISPOSITIVO, cargar


def pronosticar(panel, objetivo, ruta_pesos, L=26, H=8):
    df = D.preparar(panel, objetivo)
    n_prov = int(df["prov_idx"].max()) + 1
    modelo, ck = cargar(ruta_pesos, n_prov, L=L, H=H)
    info, variables = ck["info"], ck["variables"]
    filas = []
    for p, g in df.groupby("prov_idx"):
        x = (g[variables].to_numpy(float)[-L:] - ck["esc_media"]) / ck["esc_desv"]
        ult = g.iloc[-1]
        with torch.no_grad():
            out = modelo(torch.tensor(x[None], dtype=torch.float32, device=DISPOSITIVO),
                         torch.tensor([p], device=DISPOSITIVO)).cpu().numpy()[0]
        out = out * info["y_desv"] + info["y_media"] + (ult["y"] if info.get("residual", True) else 0)
        casos = np.expm1(np.clip(out, 0, None))
        inicio = pd.Timestamp(ult["fecha_inicio"])
        for h in range(H):
            filas.append({"ubigeo_prov": ult["ubigeo_prov"], "departamento": ult["departamento"],
                          "provincia": ult["provincia"], "semanas_adelante": h + 1,
                          "semana_inicia": (inicio + pd.Timedelta(weeks=h + 1)).date(),
                          "casos_pronosticados": float(casos[h])})
    return pd.DataFrame(filas), {"ultima_semana": f"{int(ult['ano'])}-SE{int(ult['semana']):02d}", **info}


def medicamentos_dengue(pron, panel, kit, desde=2015):
    """Reparte los casos pronosticados por gravedad (proporción observada en la provincia desde 2015)
    y los multiplica por el kit de medicamentos por caso."""
    h = panel[panel["ano"] >= desde].groupby("ubigeo_prov")[["dengue_ssa", "dengue_csa", "dengue_grave"]].sum()
    prop = h.div(h.sum(axis=1).replace(0, np.nan), axis=0).fillna({"dengue_ssa": 1, "dengue_csa": 0, "dengue_grave": 0})
    tot = pron.groupby(["ubigeo_prov", "provincia"])["casos_pronosticados"].sum().reset_index()
    tot = tot.merge(prop.reset_index(), on="ubigeo_prov")
    k = kit[kit["enfermedad"] == "Dengue"]
    for med, g in k.groupby("medicamento"):
        tot[med] = sum(tot["casos_pronosticados"] * tot[r.variable_panel] * r.cantidad_por_caso
                       for r in g.itertuples())
    return tot.drop(columns=["dengue_ssa", "dengue_csa", "dengue_grave"])
