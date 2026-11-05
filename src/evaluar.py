"""Métricas: error en casos, por horizonte y detección de semanas de alerta (brote)."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def a_casos(y_log):
    return np.expm1(np.clip(y_log, 0, None))


def metricas(y_log, p_log):
    """MAE y RMSE en casos (promedio de los H horizontes) y MAE en log1p."""
    yc, pc = a_casos(y_log), a_casos(p_log)
    return {"MAE": float(np.mean(np.abs(yc - pc))),
            "RMSE": float(np.sqrt(np.mean((yc - pc) ** 2))),
            "MAE_log": float(np.mean(np.abs(y_log - p_log)))}


def metricas_por_horizonte(y_log, p_log):
    filas = []
    for h in range(y_log.shape[1]):
        m = metricas(y_log[:, h:h + 1], p_log[:, h:h + 1])
        m["horizonte"] = h + 1
        filas.append(m)
    return pd.DataFrame(filas)


def umbrales_alerta(df, percentil=90, anios=5, minimo=1.0):
    """Umbral de alerta por provincia y año: percentil 90 de los casos semanales de los 5 años
    anteriores (idea parecida al límite superior del canal endémico que usa el MINSA)."""
    u = {}
    for p, g in df.groupby("prov_idx"):
        for a in range(int(df["ano"].min()) + 1, int(df["ano"].max()) + 1):
            h = g[(g["ano"] >= a - anios) & (g["ano"] < a)]["casos"]
            u[(int(p), a)] = max(float(h.quantile(percentil / 100)), minimo) if len(h) else minimo
    return u


def deteccion_brotes(df, v, p_log, umbrales, horizonte=4):
    """F1, precisión y sensibilidad para anticipar con 'horizonte' semanas si una semana
    superará el umbral de alerta de su provincia."""
    h = horizonte - 1
    anios = df["ano"].to_numpy()[v.origen + horizonte]
    u = np.array([umbrales.get((int(p), int(a)), 1.0) for p, a in zip(v.P, anios)])
    real = a_casos(v.Y[:, h]) > u
    pred = a_casos(p_log[:, h]) > u
    return {"F1": f1_score(real, pred, zero_division=0),
            "Precision": precision_score(real, pred, zero_division=0),
            "Sensibilidad": recall_score(real, pred, zero_division=0),
            "semanas_alerta_reales": int(real.sum())}


def resumen_semillas(filas, claves=("MAE_val", "MAE", "RMSE", "MAE_log", "F1")):
    """Tabla media ± desviación estándar por modelo a partir de una lista de dicts con 'modelo'.
    MAE_val es el error en validación; el resto son métricas del conjunto de prueba."""
    df = pd.DataFrame(filas)
    claves = [c for c in claves if c in df.columns]
    agg = df.groupby("modelo", sort=False)[list(claves)].agg(["mean", "std"])
    out = pd.DataFrame(index=agg.index)
    for c in claves:
        out[c] = [f"{m:.3f} ± {s:.3f}" if c in ("MAE_log", "F1") else f"{m:.1f} ± {s:.1f}"
                  for m, s in zip(agg[(c, "mean")], agg[(c, "std")].fillna(0))]
    return out, agg
