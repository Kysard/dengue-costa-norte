"""
Genera las figuras (PDF) y tablas (LaTeX) del artículo a partir de los resultados guardados y de los
pesos entrenados. Todo queda en paper/figuras y paper/tablas; los números del texto en paper/numeros.tex.

Uso:
  python src/articulo.py
  cd paper && latexmk -pdf Equipo5_Paper.tex
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import abastecimiento as AB
import datos as D
import evaluar as E
import experimentos as X
import lineas_base as LB
from entrenar import cargar, predecir
from interpretar import gradientes_integrados, importancia_por_grupo

RAIZ = Path(__file__).resolve().parents[1]
PAP = RAIZ / "paper"
FIG, TAB = PAP / "figuras", PAP / "tablas"
T = RAIZ / "results" / "tablas"
COL = {"Persistencia": "#8c8b86", "ARIMA + Fourier": "#d1679a", "Random forest": "#d99400", "LSTM": "#2a78d6",
       "GRU": "#e2622f", "Transformer": "#16a271", "Canal endémico (5 años)": "#6250d6", "Real": "#1f1f1e",
       "Pronóstico perfecto": "#2f8a2f"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7.5, "axes.titlesize": 8,
                     "axes.labelsize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e3e3",
                     "grid.linewidth": .5, "lines.linewidth": 1.1, "legend.frameon": False, "savefig.bbox": "tight",
                     "savefig.pad_inches": .02})
ANCHO = 3.45                       # ancho de columna IEEE en pulgadas


def guardar(fig, nombre):
    fig.savefig(FIG / f"{nombre}.pdf")
    plt.close(fig)


def media(s):
    return float(str(s).split("±")[0])


def tabla_modelos(ruta, etiqueta, caption, salida):
    t = pd.read_csv(ruta, index_col=0)
    t.index = [i.replace(" (5 años)", "") for i in t.index]
    mejor = {c: min(media(v) for v in t[c]) for c in ["MAE_val", "MAE", "RMSE", "MAE_log"] if c in t}
    mejor["F1"] = max(media(v) for v in t["F1"])
    filas = []
    for i, r in t.iterrows():
        celdas = []
        for c in ["MAE_val", "MAE", "RMSE", "MAE_log", "F1"]:
            if c not in t:
                celdas.append("--")
                continue
            v = str(r[c]).split(" ±")[0] if str(r[c]).endswith("± 0.0") or str(r[c]).endswith("± 0.000") else str(r[c]).replace(" ± ", r"{\tiny$\,\pm$").rstrip() + "}" if " ± " in str(r[c]) else str(r[c])
            celdas.append(rf"\textbf{{{v}}}" if abs(media(r[c]) - mejor[c]) < 1e-9 else v)
        sep = "\\midrule\n" if i == "LSTM" else ""
        filas.append(f"{sep}{i} & " + " & ".join(celdas) + r" \\")
    tex = (r"\begin{table}[t]" "\n" r"\centering\caption{" + caption + r"}\label{" + etiqueta + "}\n"
           r"\setlength{\tabcolsep}{2.3pt}\footnotesize" "\n"
           r"\begin{tabular}{lccccc}" "\n" r"\toprule" "\n"
           r"Modelo & MAE val. & MAE & RMSE & MAE log & F1 \\" "\n" r"\midrule" "\n" + "\n".join(filas) + "\n"
           r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    (TAB / salida).write_text(tex, encoding="utf-8")
    return t


def log_entrenamiento():
    """Resumen de cada red entrenada (épocas, pérdida de validación, tiempo) leído de los pesos guardados."""
    import torch
    filas = []
    for f in sorted((RAIZ / "results" / "pesos").glob("*.pt")):
        i = torch.load(f, map_location="cpu", weights_only=False)["info"]
        filas.append({"archivo": f.name, "objetivo": i["objetivo"], "modelo": i["modelo"], "semilla": i["semilla"], "epocas": i["epocas"],
                      "val_mse": round(float(i["val_mse"]), 5), "segundos": round(float(i["segundos"]), 1), "parametros": i["parametros"]})
    (RAIZ / "results" / "logs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(RAIZ / "results" / "logs" / "entrenamiento_redes.csv", index=False)


def main():
    log_entrenamiento()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    cfg = X.cargar_config()
    panel = X.cargar_panel()
    num = {}

    # ---------- tablas de comparación ----------
    td = tabla_modelos(T / "dengue_comparacion.csv", "tab:dengue",
                       r"Dengue: error en prueba (2021-2025) y en validación (2017-2020). Media $\pm$ d.e. de 5 semillas; "
                       r"MAE y RMSE en casos por provincia y semana, promedio de los 8 horizontes; F1 de alerta a 4 semanas.",
                       "tabla_dengue.tex")
    te = tabla_modelos(T / "eda_comparacion.csv", "tab:eda",
                       r"EDA en menores de 5 años: mismas métricas que la Tabla~\ref{tab:dengue}.", "tabla_eda.tex")
    for nom, t, pre in [("dengue", td, "D"), ("eda", te, "E")]:
        for i in t.index:
            k = {"Persistencia": "Pers", "Canal endémico": "Canal", "ARIMA + Fourier": "Arima", "Random forest": "RF",
                 "LSTM": "LSTM", "GRU": "GRU", "Transformer": "Tr"}[i]
            for c in ["MAE_val", "MAE", "RMSE", "F1"]:
                if c in t:
                    num[f"{pre}{k}{c.replace('_', '')}"] = media(t.loc[i, c])

    # ---------- ablación ----------
    a = pd.read_csv(T / "dengue_ablacion.csv", header=[0, 1], index_col=0)
    nombres = {"completo": "Completo", "sin_clima": "Sin clima", "sin_enso": "Sin El Niño",
               "solo_casos": "Solo casos y estación", "sin_residual": "Sin conexión residual"}
    filas = []
    for v in a.index:
        c = [f"{a.loc[v, (m, 'mean')]:.1f}{{\\tiny$\\,\\pm${a.loc[v, (m, 'std')]:.1f}}}" for m in ["MAE_val", "MAE", "RMSE"]]
        c.append(f"{a.loc[v, ('F1', 'mean')]:.3f}")
        filas.append(f"{nombres[v]} & " + " & ".join(c) + r" \\")
        num[f"Ab{v.replace('_', '')}MAE"] = a.loc[v, ("MAE", "mean")]
        num[f"Ab{v.replace('_', '')}RMSE"] = a.loc[v, ("RMSE", "mean")]
        num[f"Ab{v.replace('_', '')}MAEval"] = a.loc[v, ("MAE_val", "mean")]
    (TAB / "tabla_ablacion.tex").write_text(
        r"\begin{table}[t]\centering\caption{Ablación de la LSTM para dengue (5 semillas, misma partición). El modelo completo usa casos, estacionalidad, clima y El Niño con conexión residual.}\label{tab:ablacion}" "\n"
        r"\setlength{\tabcolsep}{3pt}\footnotesize\begin{tabular}{lcccc}\toprule" "\n"
        r"Variante & MAE val. & MAE & RMSE & F1 \\\midrule" "\n" + "\n".join(filas) + "\n"
        r"\bottomrule\end{tabular}\end{table}" "\n", encoding="utf-8")

    # ---------- abastecimiento ----------
    s = pd.read_csv(T / "dengue_abastecimiento.csv")
    filas = []
    for _, r in s.iterrows():
        filas.append(f"{r.politica.replace(' (5 años)', '')} & {r['nivel_servicio_%']:.1f} & {r['tratamientos_no_atendidos']:,.0f} & "
                     f"{r['semanas_con_quiebre_%']:.1f} & {r['inventario_promedio_semanal']:,.0f}" r" \\")
        k = {"Canal endémico (5 años)": "Canal", "Random forest": "RF", "LSTM": "LSTM", "Pronóstico perfecto": "Perf"}[r.politica]
        num[f"S{k}ns"], num[f"S{k}fal"], num[f"S{k}inv"], num[f"S{k}qb"] = r["nivel_servicio_%"], r["tratamientos_no_atendidos"], r["inventario_promedio_semanal"], r["semanas_con_quiebre_%"]
    (TAB / "tabla_abastecimiento.tex").write_text(
        r"\begin{table}[t]\centering\caption{Simulación de stock base para dengue en las 14 provincias, 2021-2025 "
        r"(tiempo de entrega de 4 semanas, $z=1{,}65$). Inventario en tratamientos al cierre de cada semana.}\label{tab:stock}" "\n"
        r"\setlength{\tabcolsep}{3pt}\footnotesize\begin{tabular}{lcccc}\toprule" "\n"
        r"Política & \begin{tabular}[c]{@{}c@{}}Nivel de\\servicio (\%)\end{tabular} & \begin{tabular}[c]{@{}c@{}}No\\atendidos\end{tabular} & "
        r"\begin{tabular}[c]{@{}c@{}}Semanas con\\quiebre (\%)\end{tabular} & \begin{tabular}[c]{@{}c@{}}Inventario\\promedio\end{tabular} \\\midrule" "\n"
        + "\n".join(filas).replace(",", "\\,") + "\n" r"\bottomrule\end{tabular}\end{table}" "\n", encoding="utf-8")

    # ---------- costo ----------
    c = pd.read_csv(T / "costo_computo.csv")
    c = c[c["parametros"].notna() & (c["parametros"] != "")]
    filas = []
    for m in ["LSTM", "GRU", "Transformer"]:
        d_ = c[(c["modelo"] == m) & (c["objetivo"] == "dengue")].iloc[0]
        e_ = c[(c["modelo"] == m) & (c["objetivo"] == "eda")].iloc[0]
        filas.append(f"{m} & {int(float(d_.parametros)):,} & {float(d_.epocas):.1f} & {float(d_.segundos_por_semilla):.0f} & {float(e_.segundos_por_semilla):.0f}" r" \\")
        num[f"P{m[:2]}"] = int(float(d_.parametros))
    (TAB / "tabla_costo.tex").write_text(
        r"\begin{table}[t]\centering\caption{Tamaño y costo de entrenamiento por semilla (CPU de 2 núcleos, sin GPU). Los tiempos varían con la carga de la máquina.}\label{tab:costo}" "\n"
        r"\setlength{\tabcolsep}{4pt}\footnotesize\begin{tabular}{lcccc}\toprule" "\n"
        r"Modelo & Parámetros & Épocas & \begin{tabular}[c]{@{}c@{}}Segundos\\(dengue)\end{tabular} & \begin{tabular}[c]{@{}c@{}}Segundos\\(EDA)\end{tabular} \\\midrule" "\n"
        + "\n".join(filas).replace(",", "\\,") + "\n" r"\bottomrule\end{tabular}\end{table}" "\n", encoding="utf-8")

    # ---------- figura 1: series ----------
    g = panel.groupby(["ano", "semana"]).agg(fecha=("fecha_inicio", "first"), dengue=("dengue_total", "sum"),
                                              eda=("eda_men5", "sum"), nino=("nino12_anom", "first")).reset_index()
    g["fecha"] = pd.to_datetime(g["fecha"])
    fig, ax = plt.subplots(3, 1, figsize=(7.1, 3.3), sharex=True, gridspec_kw={"height_ratios": [1.6, 1.2, .9]})
    ax[0].plot(g.fecha, g.dengue, color="#c8414f", lw=.9)
    ax[0].set_ylabel("Dengue\n(casos/sem.)")
    for anio, txt in [(2017, "Niño costero 2017"), (2023, "Ciclón Yaku y\nNiño costero 2023")]:
        i = g[g.ano == anio].dengue.idxmax()
        ax[0].annotate(txt, xy=(g.fecha[i], g.dengue[i]), xytext=(-5, -4), textcoords="offset points", ha="right", va="top",
                       fontsize=6.5, color="#444", linespacing=1.0)
    ax[1].plot(g.fecha, g.eda, color="#2f6fae", lw=.8)
    ax[1].set_ylabel("EDA $<$5 años\n(episodios/sem.)")
    ax[2].fill_between(g.fecha, g.nino, 0, where=g.nino > 0, color="#d1443c", alpha=.7, lw=0)
    ax[2].fill_between(g.fecha, g.nino, 0, where=g.nino <= 0, color="#2f6fae", alpha=.6, lw=0)
    ax[2].set_ylabel("Niño 1+2\n(°C)")
    for a_ in ax:
        a_.axvspan(pd.Timestamp("2017-01-01"), pd.Timestamp("2020-12-31"), color="#6250d6", alpha=.06, lw=0)
        a_.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31"), color="#c8414f", alpha=.06, lw=0)
    ax[1].text(pd.Timestamp("2019-01-01"), ax[1].get_ylim()[1] * .88, "validación", ha="center", fontsize=6.5, color="#6250d6")
    ax[1].text(pd.Timestamp("2023-07-01"), ax[1].get_ylim()[1] * .88, "prueba", ha="center", fontsize=6.5, color="#c8414f")
    fig.align_ylabels()
    guardar(fig, "fig_series")
    num["casosDengue"] = int(panel.dengue_total.sum())
    num["casosEDA"] = int(panel.eda_men5.sum())

    # ---------- figura: error por horizonte ----------
    h = pd.read_csv(T / "dengue_mae_por_horizonte.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(ANCHO, 2.05))
    for m in h.columns:
        ax.plot(h.index, h[m], marker="o", ms=2.6, color=COL[m], ls="--" if m == "Persistencia" else "-", label=m)
    ax.set_xlabel("Semanas de anticipación")
    ax.set_ylabel("MAE (casos por provincia y semana)")
    ax.legend(ncol=2, loc="upper left")
    guardar(fig, "fig_horizonte")
    for m, k in [("Persistencia", "Pers"), ("LSTM", "LSTM"), ("GRU", "GRU"), ("Transformer", "Tr"), ("Random forest", "RF"), ("ARIMA + Fourier", "Arima")]:
        num[f"H1{k}"], num[f"H8{k}"] = h.loc[1, m], h.loc[8, m]

    # ---------- predicciones con los pesos guardados ----------
    d = X.preparar(panel, "dengue", cfg)
    n_prov = int(d["df"]["prov_idx"].max()) + 1
    L, H = cfg["datos"]["L"], cfg["datos"]["H"]
    preds, mejor = {}, (np.inf, None)
    for arq in ["lstm", "transformer"]:
        pv, pt = [], []
        for s_ in cfg["semillas"]:
            mod, ck = cargar(RAIZ / "results" / "pesos" / f"dengue_{arq}_s{s_}.pt", n_prov, L, H)
            i_ = ck["info"]
            pv.append(predecir(mod, d["va"], i_["y_media"], i_["y_desv"], i_.get("residual", True)))
            pt.append(predecir(mod, d["te"], i_["y_media"], i_["y_desv"], i_.get("residual", True)))
            if arq == "lstm" and i_["val_mse"] < mejor[0]:
                mejor = (i_["val_mse"], mod)
        preds[arq] = (np.mean(pv, 0), np.mean(pt, 0))

    # ---------- figura: gradientes integrados ----------
    rng = np.random.default_rng(0)
    muestra = rng.choice(len(d["te"].X), size=1000, replace=False)
    attr = gradientes_integrados(mejor[1], d["te"].X[muestra], d["te"].P[muestra], salida=3, pasos=32)
    grupos = {"Historia de casos": D.F_HISTORIA[:1], "Estacionalidad": D.F_HISTORIA[1:], "Clima": D.F_CLIMA, "El Niño": D.F_ENSO}
    imp = importancia_por_grupo(attr, d["variables"], grupos)
    for k_, v_ in imp.items():
        num["IG" + {"Historia de casos": "Hist", "Estacionalidad": "Est", "Clima": "Clima", "El Niño": "Nino"}[k_]] = v_
    etq = {"y": "casos (log)", "sin_sem": "seno semana", "cos_sem": "coseno semana", "t2m_media": "temp. media", "t2m_max": "temp. máx.",
           "t2m_min": "temp. mín.", "precip_log": "lluvia", "hr_media": "humedad", "nino12_anom": "Niño 1+2", "oni": "ONI"}
    m_ = np.abs(attr).mean(0).T
    fig, ax = plt.subplots(figsize=(ANCHO, 1.95))
    im = ax.imshow(m_, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(d["variables"])), [etq[v] for v in d["variables"]])
    ax.set_xticks(range(0, L, 5), [f"$t-{L - 1 - i}$" for i in range(0, L, 5)])
    ax.set_xlabel("Semana de la ventana de entrada")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=.02)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("|atribución| media", fontsize=6.5)
    guardar(fig, "fig_ig")

    # ---------- errores: semanas de alerta y 2023 ----------
    umb = E.umbrales_alerta(d["df"], cfg["alerta"]["percentil"])
    te_ = d["te"]
    anios = d["df"]["ano"].to_numpy()[te_.origen + 4]
    u = np.array([umb.get((int(p), int(a_)), 1) for p, a_ in zip(te_.P, anios)])
    alerta = E.a_casos(te_.Y[:, 3]) > u
    pers = np.repeat(te_.ult[:, None], H, 1)
    for k_, p_ in [("Pers", pers), ("LSTM", preds["lstm"][1]), ("Tr", preds["transformer"][1])]:
        e4 = np.abs(E.a_casos(te_.Y[:, 3]) - E.a_casos(p_[:, 3]))
        num[f"Al{k_}Normal"], num[f"Al{k_}Alerta"] = e4[~alerta].mean(), e4[alerta].mean()
    num["pctAlerta"] = 100 * alerta.mean()
    # pico de 2023 en Piura
    pi_ = int(d["df"].loc[d["df"].provincia == "PIURA", "prov_idx"].iloc[0])
    idx = np.where(te_.P == pi_)[0]
    idx = idx[np.argsort(te_.origen[idx])]
    fechas = pd.to_datetime(d["df"].fecha_inicio.to_numpy()[te_.origen[idx] + 4])
    real = E.a_casos(te_.Y[idx, 3])
    j = int(np.argmax(real))
    num["picoReal"], num["picoTr"] = real[j], E.a_casos(preds["transformer"][1][idx[j], 3])
    num["picoLSTM"] = E.a_casos(preds["lstm"][1][idx[j], 3])
    num["picoFecha"] = str(fechas[j].date())

    # ARIMA solo para Piura (figura)
    vp = D.Ventanas(te_.X[idx], te_.P[idx], te_.Y[idx], te_.origen[idx], te_.split[idx], te_.ult[idx])
    p_arima = LB.sarimax_fourier(d["df"], vp)
    fig, ax = plt.subplots(figsize=(ANCHO, 1.9))
    ax.plot(fechas, real, color=COL["Real"], lw=1.4, label="Real")
    ax.plot(fechas, E.a_casos(pers[idx, 3]), color=COL["Persistencia"], ls="--", label="Persistencia")
    ax.plot(fechas, E.a_casos(p_arima[:, 3]), color=COL["ARIMA + Fourier"], label="ARIMA + Fourier")
    ax.plot(fechas, E.a_casos(preds["transformer"][1][idx, 3]), color=COL["Transformer"], label="Transformer")
    ax.plot(fechas, E.a_casos(preds["lstm"][1][idx, 3]), color=COL["LSTM"], label="LSTM")
    ax.set_ylabel("Casos por semana")
    ax.set_xlim(pd.Timestamp("2022-10-01"), pd.Timestamp("2024-09-30"))
    ax.legend(loc="upper right", ncol=1)
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%y"))
    guardar(fig, "fig_piura")

    # ---------- figura: inventario simulado en Piura ----------
    canal_va, canal_te = LB.canal_endemico(d["df"], d["va"]), LB.canal_endemico(d["df"], d["te"])
    lead, z = cfg["abastecimiento"]["lead"], cfg["abastecimiento"]["z"]
    fig, ax = plt.subplots(2, 1, figsize=(ANCHO, 2.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    for nom, pva, pte in [("Canal endémico (5 años)", canal_va, canal_te), ("LSTM", preds["lstm"][0], preds["lstm"][1]),
                          ("Pronóstico perfecto", None, te_.Y)]:
        ss = AB.stock_seguridad(d["va"], pva, lead, z) if pva is not None else {}
        orig, dem, r_ = AB.serie_provincia(te_, pte, pi_, ss.get(pi_, 0.0), lead)
        f_ = pd.to_datetime(d["df"].fecha_inicio.to_numpy()[orig + 1])
        if nom == "LSTM":
            ax[0].plot(f_, dem, color=COL["Real"], lw=1.1)
            ax[0].fill_between(f_, dem, 0, color=COL["Real"], alpha=.08, lw=0)
        ax[1].plot(f_, r_["serie_inventario"], color=COL[nom], label=nom.replace(" (5 años)", ""))
    ax[0].set_ylabel("Demanda\n(trat./sem.)")
    ax[1].set_ylabel("Inventario\n(tratamientos)")
    ax[0].legend(*ax[1].get_legend_handles_labels(), ncol=1, loc="upper right", fontsize=6, title="Inventario con:", title_fontsize=6)
    ax[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    fig.align_ylabels()
    guardar(fig, "fig_stock")

    # ---------- EDA: error por horizonte de las redes (con los pesos guardados) ----------
    de = X.preparar(panel, "eda", cfg)
    filas_h = {"Persistencia": E.metricas_por_horizonte(de["te"].Y, np.repeat(de["te"].ult[:, None], H, 1)).MAE}
    for arq, nom in [("lstm", "LSTM"), ("gru", "GRU"), ("transformer", "Transformer")]:
        pt = []
        for s_ in cfg["semillas"]:
            mod, ck = cargar(RAIZ / "results" / "pesos" / f"eda_{arq}_s{s_}.pt", n_prov, L, H)
            i_ = ck["info"]
            pt.append(predecir(mod, de["te"], i_["y_media"], i_["y_desv"], i_.get("residual", True)))
        filas_h[nom] = E.metricas_por_horizonte(de["te"].Y, np.mean(pt, 0)).MAE
    he = pd.DataFrame({k_: np.asarray(v_) for k_, v_ in filas_h.items()}, index=range(1, H + 1))
    he.index.name = "horizonte"
    he.to_csv(T / "eda_mae_por_horizonte.csv")
    for m, k in [("Persistencia", "Pers"), ("LSTM", "LSTM"), ("GRU", "GRU"), ("Transformer", "Tr")]:
        num[f"EH1{k}"], num[f"EH8{k}"] = he.loc[1, m], he.loc[8, m]

    # ---------- hardware ----------
    cc = pd.read_csv(T / "costo_computo.csv")
    redes = cc[cc["parametros"].notna() & (cc["parametros"] != "")]
    num["segRedes"] = float(redes["segundos_total"].astype(float).sum())
    num["segTodo"] = float(cc["segundos_total"].astype(float).sum())
    num["minRedes"], num["minTodo"] = num["segRedes"] / 60, num["segTodo"] / 60
    try:   # duración real del cuaderno completo (nbconvert guarda la hora de cada celda)
        celdas = json.load(open(RAIZ / "notebooks" / "Proyecto_Grupo5_Dengue_EDA.ipynb", encoding="utf-8"))["cells"]
        ts = [c["metadata"]["execution"] for c in celdas if c.get("metadata", {}).get("execution")]
        dur = pd.Timestamp(ts[-1]["shell.execute_reply"]) - pd.Timestamp(ts[0]["iopub.execute_input"])
        num["minNotebook"] = dur.total_seconds() / 60
    except (KeyError, IndexError, FileNotFoundError):
        num["minNotebook"] = num["minTodo"]
    num["nTr"], num["nVa"], num["nTe"] = len(d["tr"].X), len(d["va"].X), len(d["te"].X)
    num["filasPanel"] = len(panel)

    # ---------- macros ----------
    def fmt(v):
        if isinstance(v, str):
            return v
        if abs(v) >= 1000:
            return f"{v:,.0f}".replace(",", "\\,")
        return f"{v:.3f}" if abs(v) < 1 else f"{v:.1f}"
    # reducciones de error que se citan en el texto (en %)
    red = lambda a_, b_: 100 * (1 - a_ / b_)
    num["redTrPers"] = red(num["DTrMAE"], num["DPersMAE"])
    num["redTrArima"] = red(num["DTrMAE"], num["DArimaMAE"])
    num["redHochoTrPers"] = red(num["H8Tr"], num["H8Pers"])
    num["redHochoTrArima"] = red(num["H8Tr"], num["H8Arima"])
    num["redSfal"] = red(num["SLSTMfal"], num["SCanalfal"])
    num["redSinv"] = red(num["SLSTMinv"], num["SCanalinv"])
    mejor_e = min(["LSTM", "GRU", "Tr"], key=lambda k_: num[f"E{k_}MAE"])
    num["EMejorMAE"] = num[f"E{mejor_e}MAE"]
    num["EMejorNombre"] = {"LSTM": "LSTM", "GRU": "GRU", "Tr": "Transformer"}[mejor_e]
    num["redEPers"] = red(num["EMejorMAE"], num["EPersMAE"])
    num["redEArima"] = red(num["EMejorMAE"], num["EArimaMAE"])
    num["redERF"] = red(num["EMejorMAE"], num["ERFMAE"])
    num["redEHochoTrPers"] = red(num["EH8Tr"], num["EH8Pers"])
    digitos = dict(zip("0123456789", ["Cero", "Uno", "Dos", "Tres", "Cuatro", "Cinco", "Seis", "Siete", "Ocho", "Nueve"]))
    lineas = ["% Generado por src/articulo.py: no editar a mano"]
    for k_, v_ in num.items():
        clave = "".join(digitos.get(ch, ch) for ch in k_ if ch.isalnum())
        lineas.append(rf"\newcommand{{\n{clave}}}{{{fmt(v_)}}}")
    (PAP / "numeros.tex").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    json.dump({k_: (v_ if isinstance(v_, str) else float(v_)) for k_, v_ in num.items()}, open(PAP / "numeros.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k_: (v_ if isinstance(v_, str) else round(float(v_), 3)) for k_, v_ in num.items()}, indent=0, ensure_ascii=False))


if __name__ == "__main__":
    main()
