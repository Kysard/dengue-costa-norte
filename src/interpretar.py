"""Interpretabilidad: gradientes integrados (Integrated Gradients) sobre la entrada del modelo."""
import numpy as np
import torch

from entrenar import DISPOSITIVO


def gradientes_integrados(modelo, X, P, salida=3, pasos=32, lote=256):
    """Atribución de cada variable y semana de la ventana al pronóstico del horizonte 'salida'
    (índice 3 = 4 semanas adelante). Línea de referencia: ventana con todas las variables en su
    promedio de entrenamiento (ceros, porque están estandarizadas).
    Devuelve un arreglo (N, L, F)."""
    modelo.eval()
    out = []
    with torch.backends.cudnn.flags(enabled=False):  # el backward de RNN en cuDNN exige modo train
        for i in range(0, len(X), lote):
            x = torch.tensor(X[i:i + lote], dtype=torch.float32, device=DISPOSITIVO)
            p = torch.tensor(P[i:i + lote], dtype=torch.long, device=DISPOSITIVO)
            base = torch.zeros_like(x)
            total = torch.zeros_like(x)
            for k in range(1, pasos + 1):
                xk = (base + k / pasos * (x - base)).requires_grad_(True)
                y = modelo(xk, p)[:, salida].sum()
                g, = torch.autograd.grad(y, xk)
                total += g
            out.append(((x - base) * total / pasos).detach().cpu().numpy())
    return np.concatenate(out)


def importancia_por_grupo(attr, variables, grupos):
    """Suma de |atribución| por grupo de variables, normalizada a 100 %."""
    imp = np.abs(attr).sum(axis=(0, 1))
    tot = imp.sum()
    return {g: float(sum(imp[variables.index(v)] for v in vs if v in variables) / tot * 100)
            for g, vs in grupos.items()}
