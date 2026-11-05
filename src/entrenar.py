"""Entrenamiento con parada temprana, varias semillas y guardado de pesos."""
import random
import time

import numpy as np
import torch
import torch.nn as nn

from modelos import construir_modelo, contar_parametros

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fijar_semilla(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _objetivo(v, residual):
    """Con residual=True el modelo aprende el cambio respecto a la última semana observada."""
    return v.Y - v.ult[:, None] if residual else v.Y


def _tensores(v, y_media, y_desv, residual):
    X = torch.tensor(v.X, dtype=torch.float32)
    P = torch.tensor(v.P, dtype=torch.long)
    Y = torch.tensor((_objetivo(v, residual) - y_media) / y_desv, dtype=torch.float32)
    return X, P, Y


def predecir(modelo, v, y_media, y_desv, residual=True, lote=1024):
    """Devuelve log1p(casos) pronosticado, forma (N, H)."""
    modelo.eval()
    salida = []
    with torch.no_grad():
        for i in range(0, len(v.X), lote):
            x = torch.tensor(v.X[i:i + lote], dtype=torch.float32, device=DISPOSITIVO)
            p = torch.tensor(v.P[i:i + lote], dtype=torch.long, device=DISPOSITIVO)
            salida.append(modelo(x, p).cpu().numpy())
    out = np.concatenate(salida) * y_desv + y_media
    if residual:
        out = out + v.ult[:, None]
    return np.clip(out, 0, None)


def entrenar(nombre, v_train, v_val, n_prov, semilla=0, L=26, H=4, epocas=60, lr=1e-3, lote=256,
             paciencia=8, peso_decay=1e-4, residual=True, verbose=False, **kw_modelo):
    """Entrena un modelo y devuelve (modelo, info). La pérdida es el MSE sobre log1p(casos) estandarizado."""
    fijar_semilla(semilla)
    obj = _objetivo(v_train, residual)
    y_media, y_desv = float(obj.mean()), float(obj.std() + 1e-6)
    modelo = construir_modelo(nombre, v_train.X.shape[2], n_prov, L=L, H=H, **kw_modelo).to(DISPOSITIVO)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=peso_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    perdida = nn.MSELoss()
    Xtr, Ptr, Ytr = _tensores(v_train, y_media, y_desv, residual)
    Xva, Pva, Yva = [t.to(DISPOSITIVO) for t in _tensores(v_val, y_media, y_desv, residual)]
    g = torch.Generator().manual_seed(semilla)
    mejor, mejor_estado, sin_mejora, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(epocas):
        modelo.train()
        orden = torch.randperm(len(Xtr), generator=g)
        acum = 0.0
        for i in range(0, len(orden), lote):
            b = orden[i:i + lote]
            x, p, y = Xtr[b].to(DISPOSITIVO), Ptr[b].to(DISPOSITIVO), Ytr[b].to(DISPOSITIVO)
            opt.zero_grad()
            l = perdida(modelo(x, p), y)
            l.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            acum += l.item() * len(b)
        modelo.eval()
        with torch.no_grad():
            lv = perdida(modelo(Xva, Pva), Yva).item()
        hist.append((ep + 1, acum / len(Xtr), lv))
        sched.step(lv)
        if verbose:
            print(f"época {ep + 1:02d}  train {acum / len(Xtr):.4f}  val {lv:.4f}")
        if lv < mejor - 1e-4:
            mejor, sin_mejora = lv, 0
            mejor_estado = {k: t.detach().cpu().clone() for k, t in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    modelo.load_state_dict(mejor_estado)
    info = {"modelo": nombre, "semilla": semilla, "epocas": len(hist), "val_mse": mejor,
            "segundos": time.time() - t0, "parametros": contar_parametros(modelo),
            "y_media": y_media, "y_desv": y_desv, "residual": residual, "historial": hist,
            "kw_modelo": kw_modelo}
    return modelo, info


def guardar(modelo, info, variables, escalador, ruta):
    torch.save({"estado": modelo.state_dict(), "info": {k: v for k, v in info.items() if k != "historial"},
                "variables": variables, "esc_media": escalador.media, "esc_desv": escalador.desv}, ruta)


def cargar(ruta, n_prov, L=26, H=4):
    ck = torch.load(ruta, map_location=DISPOSITIVO, weights_only=False)
    info = ck["info"]
    m = construir_modelo(info["modelo"], len(ck["variables"]), n_prov, L=L, H=H, **info.get("kw_modelo", {}))
    m.load_state_dict(ck["estado"])
    return m.to(DISPOSITIVO), ck
