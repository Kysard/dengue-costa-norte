"""Simulación de abastecimiento con política de stock base (revisión semanal y tiempo de entrega)."""
import numpy as np
import pandas as pd

from evaluar import a_casos


def simular(demanda, pron, lead=4, ss=0.0):
    """demanda: (T,) tratamientos requeridos en las semanas 1..T.
    pron: (T, H) pronóstico hecho al cierre de la semana t-1 para las semanas t..t+H-1 (en tratamientos).
    Cada semana se pide lo necesario para que el inventario (en almacén + en camino) cubra el pronóstico
    de las próximas 'lead' semanas más un stock de seguridad. El pedido llega 'lead' semanas después."""
    T = len(demanda)
    llegadas = np.zeros(T + lead + 1)
    stock = float(pron[0, :lead].sum() + ss)        # se empieza con el nivel objetivo
    atendido, faltante, inventario = np.zeros(T), np.zeros(T), np.zeros(T)
    for t in range(T):
        stock += llegadas[t]
        atendido[t] = min(stock, demanda[t])
        faltante[t] = demanda[t] - atendido[t]
        stock -= atendido[t]
        inventario[t] = stock
        if t + 1 < T:
            en_camino = llegadas[t + 1:t + lead].sum()
            objetivo = pron[t + 1, :lead].sum() + ss
            pedido = max(0.0, objetivo - stock - en_camino)
            llegadas[t + lead] += pedido
    return {"demanda": demanda.sum(), "atendido": atendido.sum(), "faltante": faltante.sum(),
            "semanas_quiebre": int(((faltante >= 0.5) & (demanda >= 0.5)).sum()),
            "semanas_con_demanda": int((demanda >= 0.5).sum()),
            "inventario_promedio": inventario.mean(),
            "serie_inventario": inventario, "serie_faltante": faltante}


def stock_seguridad(v_val, p_val, lead=4, z=1.65):
    """Stock de seguridad por provincia = z x desviación del error del pronóstico acumulado de 'lead' semanas,
    medido en el conjunto de validación."""
    real = a_casos(v_val.Y[:, :lead]).sum(1)
    pred = a_casos(p_val[:, :lead]).sum(1)
    err = pd.Series(real - pred).groupby(v_val.P).std().fillna(0)
    return (z * err).to_dict()


def serie_provincia(v_test, p_log, prov, ss, lead=4):
    """Simulación de una provincia: devuelve (orígenes ordenados, demanda, resultado)."""
    idx = np.where(v_test.P == prov)[0]
    idx = idx[np.argsort(v_test.origen[idx])]
    demanda = a_casos(v_test.Y[idx, 0])
    return v_test.origen[idx], demanda, simular(demanda, a_casos(p_log[idx]), lead=lead, ss=ss)


def simular_politicas(v_test, pronosticos, ss_por_politica, lead=4):
    """pronosticos: dict nombre -> (N, H) en log1p para las ventanas de prueba.
    Devuelve una tabla con el nivel de servicio y el inventario promedio de cada política."""
    filas = []
    for nombre, p_log in pronosticos.items():
        tot = {"demanda": 0, "atendido": 0, "faltante": 0, "semanas_quiebre": 0, "semanas_con_demanda": 0,
               "inventario_promedio": 0}
        for p in np.unique(v_test.P):
            _, _, r = serie_provincia(v_test, p_log, p, ss_por_politica[nombre].get(int(p), 0.0), lead)
            for k in tot:
                tot[k] += r[k]
        filas.append({"politica": nombre,
                      "nivel_servicio_%": 100 * tot["atendido"] / tot["demanda"],
                      "tratamientos_no_atendidos": tot["faltante"],
                      "semanas_con_quiebre_%": 100 * tot["semanas_quiebre"] / max(tot["semanas_con_demanda"], 1),
                      "inventario_promedio_semanal": tot["inventario_promedio"]})
    return pd.DataFrame(filas)
