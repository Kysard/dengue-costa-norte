"""Arquitecturas: LSTM, GRU y Transformer para pronóstico multi-horizonte."""
import math

import torch
import torch.nn as nn


class ModeloRecurrente(nn.Module):
    """LSTM o GRU. A cada semana de la ventana se le concatena un embedding de la provincia.
    La salida es el pronóstico de log1p(casos) para las próximas H semanas."""

    def __init__(self, n_vars, n_prov, tipo="lstm", oculto=64, capas=2, dropout=0.2, H=4, emb=4):
        super().__init__()
        self.emb = nn.Embedding(n_prov, emb)
        rnn = nn.LSTM if tipo == "lstm" else nn.GRU
        self.rnn = rnn(n_vars + emb, oculto, num_layers=capas, batch_first=True,
                       dropout=dropout if capas > 1 else 0.0)
        self.cabeza = nn.Sequential(nn.Dropout(dropout), nn.Linear(oculto, H))

    def forward(self, x, p):
        e = self.emb(p).unsqueeze(1).expand(-1, x.size(1), -1)
        h, _ = self.rnn(torch.cat([x, e], dim=-1))
        return self.cabeza(h[:, -1])


class ModeloTransformer(nn.Module):
    """Encoder Transformer sobre la ventana de L semanas (atención entre semanas).
    Se usa el estado de la última semana para pronosticar las H siguientes."""

    def __init__(self, n_vars, n_prov, L=26, d=64, cabezas=4, capas=2, dropout=0.1, H=4):
        super().__init__()
        self.entrada = nn.Linear(n_vars, d)
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        nn.init.normal_(self.pos, std=0.02)
        self.emb = nn.Embedding(n_prov, d)
        capa = nn.TransformerEncoderLayer(d, cabezas, dim_feedforward=2 * d, dropout=dropout,
                                          batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(capa, capas)
        self.norm = nn.LayerNorm(d)
        self.cabeza = nn.Linear(d, H)
        self.escala = math.sqrt(d)

    def forward(self, x, p):
        z = self.entrada(x) + self.pos[:, :x.size(1)] + self.emb(p).unsqueeze(1)
        z = self.encoder(z)
        return self.cabeza(self.norm(z[:, -1]))


def construir_modelo(nombre, n_vars, n_prov, L=26, H=4, **kw):
    if nombre in ("lstm", "gru"):
        return ModeloRecurrente(n_vars, n_prov, tipo=nombre, H=H, **kw)
    if nombre == "transformer":
        return ModeloTransformer(n_vars, n_prov, L=L, H=H, **kw)
    raise ValueError(nombre)


def contar_parametros(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
