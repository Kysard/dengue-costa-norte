"""Figuras del proyecto (se guardan en results/figuras)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluar import a_casos

TINTA, TINTA2, REJILLA = "#1f1f1e", "#52514e", "#e6e5e1"
COLORES = {"LSTM": "#2a78d6", "GRU": "#eb6834", "Transformer": "#1baf7a", "Random forest": "#eda100",
           "ARIMA + Fourier": "#e87ba4", "Canal endémico (5 años)": "#6250d6", "Persistencia": "#8c8b86",
           "Real": TINTA, "Pronóstico perfecto": "#008300"}
SALIDA = Path(__file__).resolve().parents[1] / "results" / "figuras"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True, "grid.color": REJILLA,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.6, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})


def _guardar(fig, nombre):
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / f"{nombre}.png")
    return fig


def serie_y_enso(panel):
    """Dos paneles con el mismo eje de tiempo: casos semanales (dengue y EDA <5) y anomalía del Niño 1+2."""
    g = panel.groupby(["ano", "semana"]).agg(fecha=("fecha_inicio", "first"), dengue=("dengue_total", "sum"),
                                              eda=("eda_men5", "sum"), nino=("nino12_anom", "first")).reset_index()
    g["fecha"] = pd.to_datetime(g["fecha"])
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1.2]})
    ax[0].plot(g.fecha, g.dengue, color=COLORES["LSTM"])
    ax[0].set_ylabel("Casos de dengue\npor semana")
    ax[0].set_title("Costa norte (Tumbes, Piura y Lambayeque), 2000-2025", loc="left", color=TINTA)
    ax[1].plot(g.fecha, g.eda, color=COLORES["GRU"])
    ax[1].set_ylabel("Episodios de EDA\n< 5 años por semana")
    ax[2].fill_between(g.fecha, g.nino, 0, where=g.nino > 0, color="#e34948", alpha=0.6, linewidth=0, label="cálido")
    ax[2].fill_between(g.fecha, g.nino, 0, where=g.nino <= 0, color=COLORES["LSTM"], alpha=0.5, linewidth=0,
                       label="frío")
    ax[2].set_ylabel("Anomalía\nNiño 1+2 (°C)")
    ax[2].legend(loc="upper left", ncol=2)
    for anio, t in [(2017, "Niño costero 2017"), (2023, "Ciclón Yaku y Niño costero 2023")]:
        pico = g[g.ano == anio].dengue.idxmax()
        ax[0].annotate(t, xy=(g.fecha[pico], g.dengue[pico]), xytext=(-12, -4), textcoords="offset points",
                       ha="right", va="top", color=TINTA2, fontsize=9)
    ax[2].set_xlabel("Fecha de inicio de la semana epidemiológica")
    fig.align_ylabels()
    return _guardar(fig, "01_serie_casos_enso")


def estacionalidad(panel):
    g = panel[panel.ano <= 2016].groupby("semana")[["dengue_total", "eda_men5"]].mean()
    g = g / g.max()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(g.index, g.dengue_total, color=COLORES["LSTM"], label="Dengue")
    ax.plot(g.index, g.eda_men5, color=COLORES["GRU"], label="EDA < 5 años")
    ax.set_xlabel("Semana epidemiológica")
    ax.set_ylabel("Casos promedio\n(relativo al máximo)")
    ax.set_title("Estacionalidad promedio 2000-2016", loc="left")
    ax.legend()
    return _guardar(fig, "02_estacionalidad")


def correlacion_rezagos(df, variables=("precip_mm", "t2m_media", "nino12_anom"), max_rezago=16):
    """Correlación entre log(casos+1) de la semana t y cada variable climática en t - k."""
    etiquetas = {"precip_mm": "Lluvia semanal", "t2m_media": "Temperatura media", "nino12_anom": "Anomalía Niño 1+2"}
    colores = [COLORES["LSTM"], COLORES["GRU"], COLORES["Transformer"]]
    tr = df[df.split == "train"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for var, c in zip(variables, colores):
        r = []
        for k in range(max_rezago + 1):
            x = tr.groupby("prov_idx")[var].shift(k)
            m = x.notna()
            r.append(np.corrcoef(x[m], tr.loc[m, "y"])[0, 1])
        ax.plot(range(max_rezago + 1), r, marker="o", markersize=4, color=c, label=etiquetas[var])
    ax.axhline(0, color=TINTA2, lw=0.8)
    ax.set_xlabel("Rezago k (semanas antes)")
    ax.set_ylabel("Correlación con\nlog(casos + 1)")
    ax.set_title("¿Con cuánta anticipación se relaciona el clima con el dengue? (entrenamiento)", loc="left")
    ax.legend()
    return _guardar(fig, "03_correlacion_rezagos")


def curvas_entrenamiento(infos, objetivo):
    fig, ax = plt.subplots(1, len(infos), figsize=(4 * len(infos), 3.2), sharey=True)
    for a, info in zip(np.atleast_1d(ax), infos):
        h = np.array(info["historial"])
        a.plot(h[:, 0], h[:, 1], color=COLORES[info["etiqueta"]], label="entrenamiento")
        a.plot(h[:, 0], h[:, 2], color=COLORES[info["etiqueta"]], ls="--", label="validación")
        a.set_title(info["etiqueta"], loc="left")
        a.set_xlabel("Época")
        a.legend()
    np.atleast_1d(ax)[0].set_ylabel("MSE (log1p estandarizado)")
    return _guardar(fig, f"{objetivo}_curvas_entrenamiento")


def error_por_horizonte(tablas, objetivo, titulo=None):
    """tablas: dict nombre -> DataFrame de metricas_por_horizonte."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for nombre, t in tablas.items():
        ls = "--" if nombre in ("Persistencia", "Canal endémico (5 años)") else "-"
        ax.plot(t.horizonte, t.MAE, marker="o", markersize=4, ls=ls, color=COLORES[nombre], label=nombre)
    ax.set_xlabel("Semanas de anticipación del pronóstico")
    ax.set_ylabel("MAE (casos por provincia\ny semana)")
    ax.set_title(f"Error en el conjunto de prueba (2021-2025): {titulo or objetivo}", loc="left")
    ax.legend(ncol=2, fontsize=9)
    return _guardar(fig, f"{objetivo}_error_por_horizonte")


def ablacion(tabla, objetivo):
    """tabla: DataFrame con columnas MAE/RMSE en MultiIndex (mean, std), índice variante."""
    nombres = {"completo": "Completo", "sin_clima": "Sin\nclima", "sin_enso": "Sin\nEl Niño",
               "solo_casos": "Solo\ncasos", "sin_residual": "Sin conexión\nresidual"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, met in zip(ax, ["MAE", "RMSE"]):
        m, s = tabla[(met, "mean")], tabla[(met, "std")].fillna(0)
        x = np.arange(len(m))
        a.bar(x, m, yerr=s, color=[COLORES["LSTM"]] + ["#9ec1ec"] * (len(m) - 1), width=0.6,
              error_kw=dict(ecolor=TINTA2, lw=1, capsize=3))
        a.set_xticks(x, [nombres.get(i, i) for i in m.index], fontsize=9)
        a.set_ylabel(f"{met} (casos)")
        a.set_title(f"{met} en prueba, media ± d.e. de las semillas", loc="left", fontsize=10)
        a.grid(axis="x", visible=False)
        a.set_axisbelow(True)
    return _guardar(fig, f"{objetivo}_ablacion")


def mapa_atribuciones(attr, variables, objetivo):
    etiquetas = {"y": "casos (log)", "sin_sem": "seno semana", "cos_sem": "coseno semana",
                 "t2m_media": "temp. media", "t2m_max": "temp. máxima", "t2m_min": "temp. mínima",
                 "precip_log": "lluvia", "hr_media": "humedad", "nino12_anom": "Niño 1+2", "oni": "ONI"}
    imp = np.abs(attr).mean(0).T                      # (F, L)
    L = imp.shape[1]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(imp, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(variables)), [etiquetas.get(v, v) for v in variables])
    ax.set_xticks(range(0, L, 5), [f"t-{L - 1 - i}" for i in range(0, L, 5)])
    ax.set_xlabel("Semana de la ventana de entrada (t = última semana observada)")
    ax.set_title("Gradientes integrados: |atribución| media al pronóstico a 4 semanas", loc="left")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="|atribución| media")
    return _guardar(fig, f"{objetivo}_gradientes_integrados")


def error_por_provincia(tabla, objetivo):
    """tabla: índice provincia, columnas = modelos con MAE."""
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(tabla))
    w = 0.8 / len(tabla.columns)
    for k, c in enumerate(tabla.columns):
        ax.bar(x + k * w - 0.4 + w / 2, tabla[c], width=w * 0.92, color=COLORES[c], label=c)
    ax.set_xticks(x, [p.title() for p in tabla.index], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("MAE (casos por semana)")
    ax.set_title("Error en prueba por provincia (horizontes 1-8)", loc="left")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.legend()
    return _guardar(fig, f"{objetivo}_error_por_provincia")


def pronostico_provincia(fechas, real, pronos, provincia, h, objetivo):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(fechas, real, color=TINTA, lw=2, label="Real")
    for nombre, p in pronos.items():
        ax.plot(fechas, p, color=COLORES[nombre], ls="--" if nombre == "Persistencia" else "-", label=nombre)
    ax.set_ylabel("Casos por semana")
    ax.set_title(f"{provincia.title()}: pronóstico hecho {h} semanas antes vs. casos reales", loc="left")
    ax.legend()
    return _guardar(fig, f"{objetivo}_pronostico_{provincia.lower()}")


def simulacion_stock(fechas, demanda, inventarios, provincia):
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    ax[0].plot(fechas, demanda, color=TINTA, lw=2, label="Demanda (casos)")
    ax[0].set_ylabel("Tratamientos\npor semana")
    ax[0].legend()
    for nombre, inv in inventarios.items():
        ax[1].plot(fechas, inv, color=COLORES[nombre], label=nombre)
    ax[1].set_ylabel("Inventario al cierre\n(tratamientos)")
    ax[1].legend(ncol=3, fontsize=9)
    ax[0].set_title(f"{provincia.title()}: demanda e inventario simulado con cada política", loc="left")
    return _guardar(fig, f"dengue_stock_{provincia.lower()}")
