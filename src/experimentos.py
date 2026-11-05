"""
Corre todos los experimentos del artículo y guarda tablas, figuras, predicciones y pesos.

Uso (desde la carpeta del proyecto):
  python src/experimentos.py --objetivo dengue
  python src/experimentos.py --objetivo eda
  python src/experimentos.py --objetivo dengue --rapido     # 1 semilla, para probar
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import abastecimiento as ab
import datos as D
import evaluar as E
import lineas_base as LB
from entrenar import DISPOSITIVO, entrenar, guardar, predecir

RAIZ = Path(__file__).resolve().parents[1]
NOMBRES = {"persistencia": "Persistencia", "canal_endemico": "Canal endémico (5 años)",
           "sarimax": "ARIMA + Fourier", "random_forest": "Random forest",
           "lstm": "LSTM", "gru": "GRU", "transformer": "Transformer"}


def cargar_config(ruta=None):
    return yaml.safe_load(open(ruta or RAIZ / "configs" / "experimentos.yaml", encoding="utf-8"))


def cargar_panel(ruta=None):
    return pd.read_csv(ruta or RAIZ / "data" / "processed" / "panel_provincia_semana.csv",
                       dtype={"ubigeo_prov": str})


def preparar(panel, objetivo, cfg, clima=True, enso=True):
    df = D.preparar(panel, objetivo)
    variables = D.lista_variables(clima, enso)
    v, esc = D.crear_ventanas(df, variables, L=cfg["datos"]["L"], H=cfg["datos"]["H"])
    tr, va, te = (D.subconjunto(v, s) for s in ("train", "val", "test"))
    return {"df": df, "variables": variables, "esc": esc, "tr": tr, "va": va, "te": te}


def evaluar_pred(nombre, d, p_test, umbrales, cfg, semilla=0, p_val=None):
    fila = {"modelo": nombre, "semilla": semilla, **E.metricas(d["te"].Y, p_test)}
    if p_val is not None:
        fila["MAE_val"] = E.metricas(d["va"].Y, p_val)["MAE"]
    fila.update(E.deteccion_brotes(d["df"], d["te"], p_test, umbrales, cfg["alerta"]["horizonte"]))
    return fila


def correr_lineas_base(d, cfg, umbrales, semillas):
    """Devuelve filas de métricas y predicciones {nombre: (val, test)}."""
    filas, preds, tiempos = [], {}, {}
    t = time.time()
    preds["persistencia"] = (LB.persistencia(d["df"], d["va"]), LB.persistencia(d["df"], d["te"]))
    tiempos["persistencia"] = time.time() - t
    t = time.time()
    preds["canal_endemico"] = (LB.canal_endemico(d["df"], d["va"]), LB.canal_endemico(d["df"], d["te"]))
    tiempos["canal_endemico"] = time.time() - t
    t = time.time()
    preds["sarimax"] = (LB.sarimax_fourier(d["df"], d["va"]), LB.sarimax_fourier(d["df"], d["te"]))
    tiempos["sarimax"] = time.time() - t
    for k in ("persistencia", "canal_endemico", "sarimax"):
        filas.append(evaluar_pred(NOMBRES[k], d, preds[k][1], umbrales, cfg, p_val=preds[k][0]))
    t = time.time()
    pv, pt = [], []
    for s in semillas:
        _, (a, b) = LB.random_forest(d["tr"], [d["va"], d["te"]], semilla=s, n_arboles=cfg["random_forest"]["arboles"])
        pv.append(a)
        pt.append(b)
        filas.append(evaluar_pred(NOMBRES["random_forest"], d, b, umbrales, cfg, s, p_val=a))
    tiempos["random_forest"] = time.time() - t
    preds["random_forest"] = (np.mean(pv, 0), np.mean(pt, 0))
    return filas, preds, tiempos


def correr_redes(d, cfg, umbrales, semillas, modelos, objetivo, etiqueta=None, residual=True,
                 carpeta_pesos=None):
    """Entrena cada modelo con cada semilla. Devuelve filas de métricas, predicciones promedio de las
    semillas {nombre: (val, test)}, info de entrenamiento y el mejor modelo (menor pérdida de validación)."""
    filas, preds, infos, mejores = [], {}, [], {}
    n_prov = int(d["df"]["prov_idx"].max()) + 1
    for nombre in modelos:
        pv, pt, mejor = [], [], (np.inf, None, None)
        for s in semillas:
            m, info = entrenar(nombre, d["tr"], d["va"], n_prov, semilla=s, L=cfg["datos"]["L"],
                               H=cfg["datos"]["H"], residual=residual, **cfg["entrenamiento"],
                               **cfg["modelos"][nombre])
            a = predecir(m, d["va"], info["y_media"], info["y_desv"], residual)
            b = predecir(m, d["te"], info["y_media"], info["y_desv"], residual)
            pv.append(a)
            pt.append(b)
            et = etiqueta or NOMBRES[nombre]
            filas.append(evaluar_pred(et, d, b, umbrales, cfg, s, p_val=a))
            info["objetivo"], info["etiqueta"] = objetivo, et
            infos.append(info)
            if carpeta_pesos:
                Path(carpeta_pesos).mkdir(parents=True, exist_ok=True)
                guardar(m, info, d["variables"], d["esc"], Path(carpeta_pesos) / f"{objetivo}_{nombre}_s{s}.pt")
            if info["val_mse"] < mejor[0]:
                mejor = (info["val_mse"], m, info)
        preds[nombre] = (np.mean(pv, 0), np.mean(pt, 0))
        mejores[nombre] = mejor
    return filas, preds, infos, mejores


def correr_ablaciones(panel, objetivo, cfg, semillas, filas_completo, carpeta_pesos=None):
    filas = []
    for nombre, ab_cfg in cfg["ablaciones"].items():
        if nombre == "completo":
            filas += [{**f, "variante": "completo"} for f in filas_completo]
            continue
        d = preparar(panel, objetivo, cfg, clima=ab_cfg["clima"], enso=ab_cfg["enso"])
        u = E.umbrales_alerta(d["df"], cfg["alerta"]["percentil"])
        f, _, _, _ = correr_redes(d, cfg, u, semillas, ["lstm"], objetivo, etiqueta="LSTM",
                                  residual=ab_cfg["residual"])
        filas += [{**x, "variante": nombre} for x in f]
    return filas


def costo_computo(infos, tiempos_base):
    df = pd.DataFrame([{"modelo": i["etiqueta"], "segundos": i["segundos"], "epocas": i["epocas"],
                        "parametros": i["parametros"]} for i in infos])
    t = df.groupby("modelo").agg(segundos_por_semilla=("segundos", "mean"), epocas=("epocas", "mean"),
                                 parametros=("parametros", "first"), segundos_total=("segundos", "sum"))
    for k, v in tiempos_base.items():
        t.loc[NOMBRES[k], "segundos_total"] = v
    return t


def info_hardware():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sin GPU"
    return {"dispositivo": str(DISPOSITIVO), "gpu": gpu, "procesador": platform.processor() or platform.machine(),
            "python": platform.python_version(), "torch": torch.__version__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objetivo", choices=["dengue", "eda"], default="dengue")
    ap.add_argument("--rapido", action="store_true", help="una sola semilla")
    args = ap.parse_args()
    cfg = cargar_config()
    semillas = cfg["semillas"][:1] if args.rapido else cfg["semillas"]
    out_t, out_p = RAIZ / "results" / "tablas", RAIZ / "results" / "predicciones"
    out_t.mkdir(parents=True, exist_ok=True)
    out_p.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    panel = cargar_panel()
    d = preparar(panel, args.objetivo, cfg)
    u = E.umbrales_alerta(d["df"], cfg["alerta"]["percentil"])
    fb, pb, tb = correr_lineas_base(d, cfg, u, semillas)
    fr, pr, infos, _ = correr_redes(d, cfg, u, semillas, list(cfg["modelos"]), args.objetivo,
                                    carpeta_pesos=RAIZ / "results" / "pesos")
    tabla, _ = E.resumen_semillas(fb + fr)
    tabla.to_csv(out_t / f"{args.objetivo}_comparacion.csv")
    pd.DataFrame(fb + fr).to_csv(out_t / f"{args.objetivo}_comparacion_por_semilla.csv", index=False)
    print(tabla)

    lstm_filas = [f for f in fr if f["modelo"] == "LSTM"]
    fa = correr_ablaciones(panel, args.objetivo, cfg, semillas, lstm_filas)
    ta = pd.DataFrame(fa).groupby("variante", sort=False)[["MAE_val", "MAE", "RMSE", "MAE_log", "F1"]].agg(["mean", "std"])
    ta.to_csv(out_t / f"{args.objetivo}_ablacion.csv")
    print(ta)

    todos = {**pb, **pr}
    np.savez_compressed(out_p / f"{args.objetivo}.npz", Y_test=d["te"].Y, P_test=d["te"].P,
                        origen_test=d["te"].origen, **{f"{k}_test": v[1] for k, v in todos.items()},
                        **{f"{k}_val": v[0] for k, v in todos.items()})
    if args.objetivo == "dengue":
        lead, z = cfg["abastecimiento"]["lead"], cfg["abastecimiento"]["z"]
        pol = {"Canal endémico (5 años)": "canal_endemico", "Random forest": "random_forest", "LSTM": "lstm"}
        pron = {k: todos[v][1] for k, v in pol.items()}
        ss = {k: ab.stock_seguridad(d["va"], todos[v][0], lead, z) for k, v in pol.items()}
        pron["Pronóstico perfecto"] = d["te"].Y
        ss["Pronóstico perfecto"] = {}
        sim = ab.simular_politicas(d["te"], pron, ss, lead)
        sim.to_csv(out_t / "dengue_abastecimiento.csv", index=False)
        print(sim)
    costo_computo(infos, tb).to_csv(out_t / f"{args.objetivo}_costo_computo.csv")
    json.dump({**info_hardware(), "minutos_totales": (time.time() - t0) / 60},
              open(out_t / f"{args.objetivo}_hardware.json", "w"), indent=2)


if __name__ == "__main__":
    main()
