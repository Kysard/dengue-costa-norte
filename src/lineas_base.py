"""Líneas base: persistencia, canal endémico, SARIMAX con Fourier y random forest."""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def persistencia(df, v):
    """El pronóstico de las próximas H semanas es el valor de la última semana observada."""
    ult = df.loc[v.origen, "y"].to_numpy()
    return np.repeat(ult[:, None], v.Y.shape[1], axis=1)


def canal_endemico(df, v, anios=5):
    """Mediana de la misma semana epidemiológica en los 5 años anteriores (es la línea central del
    canal endémico que usa el MINSA). Es lo que se usaría para planificar 'según lo histórico'."""
    tabla = df.pivot_table(index=["prov_idx", "semana"], columns="ano", values="y")
    H = v.Y.shape[1]
    sem = df["semana"].to_numpy()
    ano = df["ano"].to_numpy()
    pred = np.zeros_like(v.Y)
    for i, o in enumerate(v.origen):
        p = int(v.P[i])
        for h in range(H):
            j = o + 1 + h
            a, s = int(ano[j]), int(min(sem[j], 52))
            cols = [c for c in range(a - anios, a) if c in tabla.columns]
            pred[i, h] = np.nanmedian(tabla.loc[(p, s), cols].to_numpy()) if cols else 0.0
    return pred


def random_forest(v_train, v_eval_list, semilla=0, n_arboles=200, residual=True):
    """Random forest multi-salida con las últimas 8 semanas de todas las variables + provincia.
    Igual que las redes, aprende el cambio respecto a la última semana (residual=True)."""
    n_prov = int(max(v.P.max() for v in [v_train] + v_eval_list)) + 1
    def plano(v):
        oh = np.eye(n_prov)[v.P]
        return np.hstack([v.X[:, -8:, :].reshape(len(v.X), -1), oh])
    rf = RandomForestRegressor(n_estimators=n_arboles, min_samples_leaf=5, max_features=0.5,
                               n_jobs=-1, random_state=semilla)
    base = (lambda v: v.ult[:, None]) if residual else (lambda v: 0.0)
    rf.fit(plano(v_train), v_train.Y - base(v_train))
    return rf, [np.clip(rf.predict(plano(v)) + base(v), 0, None) for v in v_eval_list]


def _fourier(semanas, K=3, periodo=52.18):
    t = np.asarray(semanas, dtype=float) - 1
    cols = []
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * t / periodo), np.cos(2 * np.pi * k * t / periodo)]
    return np.column_stack(cols)


def sarimax_fourier(df, v_eval, orden=(2, 0, 1), K=3):
    """ARIMA(2,0,1) sobre log1p(casos) con estacionalidad anual por términos de Fourier.
    Se ajusta por provincia SOLO con el periodo de entrenamiento; luego se pronostica desde cada
    semana de origen con los parámetros fijos (sin volver a ajustar)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    H = v_eval.Y.shape[1]
    pred = np.zeros_like(v_eval.Y)
    for p in np.unique(v_eval.P):
        g = df[df["prov_idx"] == p]
        y = g["y"].to_numpy()
        ex = _fourier(g["semana"].to_numpy(), K)
        n_tr = int((g["split"] == "train").sum())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(y[:n_tr], exog=ex[:n_tr], order=orden, trend="c").fit(disp=False, maxiter=200)
            pos = {o: k for k, o in enumerate(g.index)}
            for i in np.where(v_eval.P == p)[0]:
                t = pos[v_eval.origen[i]]
                r = res.apply(y[:t + 1], exog=ex[:t + 1])
                pred[i] = r.forecast(H, exog=ex[t + 1:t + 1 + H])
    return np.clip(pred, 0, None)
