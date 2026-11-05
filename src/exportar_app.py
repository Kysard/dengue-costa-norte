"""
Arma el aplicativo web (app/index.html): exporta los pesos entrenados de LSTM, GRU y Transformer para que
corran en el navegador, junto con las series, el clima normal de cada semana, los intervalos de error
medidos en validación y los datos para calcular pedidos de medicamentos.

Uso:
  python src/exportar_app.py                 # exporta todo y arma app/index.html
  python src/exportar_app.py --solo-html     # solo vuelve a armar el HTML con app/datos_app.json
  python src/exportar_app.py --prueba        # además guarda casos de prueba para comparar con JavaScript
"""
import argparse
import base64
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import construir_dataset as cd
import datos as D
import evaluar as E
import experimentos as X
from entrenar import cargar, predecir

RAIZ = Path(__file__).resolve().parents[1]
APP = RAIZ / "app"
ARQS = ["lstm", "gru", "transformer"]
VARS_CLIMA = ["t2m_media", "t2m_max", "t2m_min", "precip_mm", "hr_media"]


def r(a, n=2):
    return np.round(np.asarray(a, dtype=float), n).tolist()


def calendario_extendido(panel, extra=110):
    """Calendario epidemiológico del panel más 'extra' semanas futuras."""
    g = panel[panel["ubigeo_prov"] == panel["ubigeo_prov"].min()]
    ano, se, fecha = g["ano"].tolist(), g["semana"].tolist(), g["fecha_inicio"].astype(str).tolist()
    a, s = ano[-1], se[-1]
    for _ in range(extra):
        s += 1
        if s > cd.semanas_del_anio(a):
            a, s = a + 1, 1
        ano.append(a)
        se.append(s)
        fecha.append(str(cd.inicio_se1(a) + timedelta(weeks=s - 1)))
    return ano, se, fecha


def exportar_red(ruta, n_prov, cfg, tipo):
    mod, ck = cargar(ruta, n_prov, cfg["datos"]["L"], cfg["datos"]["H"])
    info = ck["info"]
    tensores = [(k, v.detach().cpu().numpy().astype(np.float32)) for k, v in mod.state_dict().items()]
    plano = np.concatenate([t.ravel() for _, t in tensores])
    datos = plano.astype(np.float16 if tipo == "f16" else np.float32).tobytes()
    esquema = [[k, list(t.shape)] for k, t in tensores]
    return mod, info, {"arq": info["modelo"], "w": base64.b64encode(datos).decode(),
                       "media": r(ck["esc_media"], 6), "desv": r(ck["esc_desv"], 6),
                       "y_media": float(info["y_media"]), "y_desv": float(info["y_desv"]),
                       "residual": bool(info.get("residual", True))}, esquema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-html", action="store_true")
    ap.add_argument("--prueba", action="store_true")
    ap.add_argument("--tipo", default="f16", choices=["f16", "f32"])
    ap.add_argument("--pesos", default=str(RAIZ / "results" / "pesos"))
    args = ap.parse_args()
    if args.solo_html:
        armar_html((APP / "datos_app.json").read_text(encoding="utf-8"))
        return

    cfg = X.cargar_config()
    panel = X.cargar_panel()
    prov = pd.read_csv(RAIZ / "data" / "external" / "provincias.csv", dtype={"ubigeo_prov": str})
    pob = pd.read_csv(RAIZ / "data" / "external" / "poblacion_censos.csv", dtype={"ubigeo_prov": str})
    kit = pd.read_csv(RAIZ / "data" / "external" / "kit_medicamentos.csv")
    L, H = cfg["datos"]["L"], cfg["datos"]["H"]
    variables = D.lista_variables()

    base = D.preparar(panel, "dengue")            # variables comunes (clima y El Niño ya rezagado)
    codigos = sorted(base["ubigeo_prov"].unique())
    n_prov = len(codigos)
    NS = len(base) // n_prov
    ano, se, fecha = calendario_extendido(panel)
    g0 = base[base["ubigeo_prov"] == codigos[0]]
    salida = {"L": L, "H": H, "NS": NS, "variables": variables, "tipo": args.tipo,
              "cal": {"ano": ano, "se": se, "fecha": fecha},
              "enso": {"nino12": r(g0["nino12_anom"]), "oni": r(g0["oni"])},
              "provincias": [], "serie": {}, "clima_normal": {}, "redes": {}, "esquema": {},
              "banda": {}, "sigma": {}, "metricas": {}, "kit": kit.to_dict(orient="records")}

    for i, c in enumerate(codigos):
        g = base[base["ubigeo_prov"] == c].sort_values(["ano", "semana"])
        p = prov[prov["ubigeo_prov"] == c].iloc[0]
        q = pob[pob["ubigeo_prov"] == c]
        pobl = cd.poblacion_anual(q, range(2000, 2031)).set_index("ano")["poblacion"]
        desde = g["ano"] >= 2015
        sev = g.loc[desde, ["dengue_ssa", "dengue_csa", "dengue_grave"]].sum()
        hosp = g.loc[desde, "eda_hosp_men5"].sum() / max(g.loc[desde, "eda_men5"].sum(), 1)
        salida["provincias"].append({
            "idx": i, "id": c, "nombre": cd.corregir_texto(pd.Series([p.provincia]))[0].title(),
            "dep": p.departamento.title(), "capital": p.capital,
            "pob": {int(a): int(v) for a, v in pobl.items()},
            "gravedad": {"dengue_ssa": float(sev.iloc[0] / max(sev.sum(), 1)),
                         "dengue_csa": float(sev.iloc[1] / max(sev.sum(), 1)),
                         "dengue_grave": float(sev.iloc[2] / max(sev.sum(), 1))},
            "hosp_eda": float(hosp)})
        salida["serie"][c] = {"dengue": g["dengue_total"].astype(int).tolist(), "eda": g["eda_men5"].astype(int).tolist(),
                              "t2m_media": r(g["t2m_media"]), "t2m_max": r(g["t2m_max"]), "t2m_min": r(g["t2m_min"]),
                              "precip_mm": r(g["precip_mm"]), "hr_media": r(g["hr_media"])}
        hist = g[g["ano"] <= 2020]                 # clima normal con entrenamiento y validación
        normal = hist.groupby("semana")[VARS_CLIMA].mean()
        normal.loc[53] = normal.loc[53] if 53 in normal.index else normal.loc[52]
        salida["clima_normal"][c] = {v: r(normal.sort_index()[v].reindex(range(1, 54)).ffill()) for v in VARS_CLIMA}

    prueba = {}
    for objetivo in ["dengue", "eda"]:
        print("exportando", objetivo)
        d = X.preparar(panel, objetivo, cfg)
        preds = {"va": {}, "te": {}}
        salida["redes"][objetivo] = {}
        for arq in ARQS:
            pv, pt, lista = [], [], []
            for s in cfg["semillas"]:
                mod, info, ex, esquema = exportar_red(Path(args.pesos) / f"{objetivo}_{arq}_s{s}.pt", n_prov, cfg, args.tipo)
                salida["esquema"][arq] = esquema
                lista.append(ex)
                va, te = d["va"], d["te"]
                esc = D.Escalador(np.array(ex["media"]), np.array(ex["desv"]))
                if np.abs(esc.media - d["esc"].media).max() > 1e-4:   # pesos entrenados con otro escalado
                    v, _ = D.crear_ventanas(d["df"], variables, L=L, H=H, escalador=esc)
                    va, te = D.subconjunto(v, "val"), D.subconjunto(v, "test")
                pv.append(predecir(mod, va, info["y_media"], info["y_desv"], info.get("residual", True)))
                pt.append(predecir(mod, te, info["y_media"], info["y_desv"], info.get("residual", True)))
                if args.prueba and s == 0:
                    rng = np.random.default_rng(1)
                    idx = rng.choice(len(d["te"].X), 12, replace=False)
                    prueba.setdefault(objetivo, {})[arq] = [{"prov": int(d["te"].P[k]),
                                                             "origen": int(d["te"].origen[k] - d["te"].P[k] * NS),
                                                             "pred": r(pt[-1][k], 6)} for k in idx]
            salida["redes"][objetivo][arq] = lista
            preds["va"][arq], preds["te"][arq] = np.mean(pv, 0), np.mean(pt, 0)
        preds["va"]["promedio"] = np.mean([preds["va"][a] for a in ARQS], 0)
        preds["te"]["promedio"] = np.mean([preds["te"][a] for a in ARQS], 0)

        # intervalo del 80 % por horizonte: cuantiles del error (en log) en validación
        salida["banda"][objetivo] = {}
        salida["sigma"][objetivo] = {}
        for k, pv in preds["va"].items():
            res = d["va"].Y - pv
            salida["banda"][objetivo][k] = {"q10": r(np.quantile(res, .1, axis=0), 4), "q90": r(np.quantile(res, .9, axis=0), 4)}
            # desviación del error acumulado de 1..8 semanas por provincia (para el stock de seguridad)
            real, pron = E.a_casos(d["va"].Y), E.a_casos(pv)
            sig = []
            for pi in range(n_prov):
                m = d["va"].P == pi
                sig.append([float(np.std(real[m, :l].sum(1) - pron[m, :l].sum(1), ddof=1)) for l in range(1, H + 1)])
            salida["sigma"][objetivo][k] = r(sig, 2)

        # error por horizonte en prueba (2021-2025)
        met = {"Persistencia": E.metricas_por_horizonte(d["te"].Y, np.repeat(d["te"].ult[:, None], H, 1)).MAE}
        for k, nom in [("lstm", "LSTM"), ("gru", "GRU"), ("transformer", "Transformer"), ("promedio", "Promedio de las tres")]:
            met[nom] = E.metricas_por_horizonte(d["te"].Y, preds["te"][k]).MAE
        salida["metricas"][objetivo] = {k: r(v, 1) for k, v in met.items()}

    datos_js = json.dumps(salida, ensure_ascii=False, separators=(",", ":"))
    (APP / "datos_app.json").write_text(datos_js, encoding="utf-8")
    if args.prueba:
        (APP.parent / "results" / "prueba_app.json").write_text(json.dumps(prueba), encoding="utf-8")
    armar_html(datos_js)


def armar_html(datos_js):
    mapa = json.loads((APP / "mapa.json").read_text(encoding="utf-8"))
    html = (APP / "plantilla.html").read_text(encoding="utf-8")
    modelo = (APP / "modelo.js").read_text(encoding="utf-8")
    html = (html.replace("/*__MODELO__*/", modelo)
                .replace("/*__DATOS__*/null", datos_js)
                .replace("/*__MAPA__*/null", json.dumps(mapa, ensure_ascii=False, separators=(",", ":"))))
    # para abrirlo con doble clic el archivo necesita declarar UTF-8 (si no, Windows muestra mal las tildes)
    cabecera = ('<!doctype html>\n<html lang="es">\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n')
    (APP / "index.html").write_text(cabecera + html, encoding="utf-8")
    print(f"app/index.html listo ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
