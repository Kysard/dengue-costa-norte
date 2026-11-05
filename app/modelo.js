/* Inferencia de los modelos entrenados (LSTM, GRU y Transformer) en el navegador.
   Reproduce exactamente las capas de PyTorch usadas en src/modelos.py, con los pesos exportados por
   src/exportar_app.py. Funciona en el navegador (variable global Modelos) y en Node (para las pruebas). */
(function (raiz) {
  "use strict";

  // ---------- decodificación de pesos (base64 -> Float32Array) ----------
  const TABLA16 = (() => {
    const t = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      t[h] = e === 0 ? s * Math.pow(2, -14) * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return t;
  })();

  function base64ABytes(b64) {
    if (typeof atob === "function") {
      const bin = atob(b64), n = bin.length, out = new Uint8Array(n);
      for (let i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
      return out;
    }
    return new Uint8Array(Buffer.from(b64, "base64"));
  }

  function decodificar(b64, tipo) {
    const bytes = base64ABytes(b64);
    if (tipo === "f32") return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2), out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) out[i] = TABLA16[u16[i]];
    return out;
  }

  /* Convierte un checkpoint exportado en un objeto {nombre_del_tensor: Float32Array} */
  function cargarRed(ck, esquema, tipo) {
    const plano = decodificar(ck.w, tipo), W = {};
    let off = 0;
    for (const [nombre, forma] of esquema) {
      const n = forma.reduce((a, b) => a * b, 1);
      W[nombre] = plano.subarray(off, off + n);
      off += n;
    }
    return {W, arq: ck.arq, media: Float32Array.from(ck.media), desv: Float32Array.from(ck.desv),
            yMedia: ck.y_media, yDesv: ck.y_desv, residual: ck.residual};
  }

  const sig = x => 1 / (1 + Math.exp(-x));

  // y = W x + b  (W: filas x columnas, guardada por filas como en PyTorch)
  function lineal(W, b, x, xo, nIn, nOut, y, yo) {
    for (let r = 0; r < nOut; r++) {
      let s = b ? b[r] : 0, wo = r * nIn;
      for (let c = 0; c < nIn; c++) s += W[wo + c] * x[xo + c];
      y[yo + r] = s;
    }
  }

  // ---------- LSTM / GRU de 2 capas con embedding de provincia ----------
  function recurrente(red, x, L, F, p) {
    const W = red.W, H = W["rnn.bias_hh_l0"].length / (red.arq === "lstm" ? 4 : 3);
    const G = red.arq === "lstm" ? 4 * H : 3 * H, E = W["rnn.weight_ih_l0"].length / G - F;
    let entrada = new Float32Array(L * (F + E)), nIn = F + E;
    for (let t = 0; t < L; t++) {
      for (let f = 0; f < F; f++) entrada[t * nIn + f] = x[t * F + f];
      for (let e = 0; e < E; e++) entrada[t * nIn + F + e] = W["emb.weight"][p * E + e];
    }
    const gi = new Float32Array(G), gh = new Float32Array(G);
    let salida = null;
    for (let capa = 0; capa < 2; capa++) {
      const Wih = W[`rnn.weight_ih_l${capa}`], Whh = W[`rnn.weight_hh_l${capa}`];
      const bih = W[`rnn.bias_ih_l${capa}`], bhh = W[`rnn.bias_hh_l${capa}`];
      const h = new Float32Array(H), c = new Float32Array(H);
      salida = new Float32Array(L * H);
      for (let t = 0; t < L; t++) {
        lineal(Wih, bih, entrada, t * nIn, nIn, G, gi, 0);
        lineal(Whh, bhh, h, 0, H, G, gh, 0);
        if (red.arq === "lstm") {            // compuertas i, f, g, o
          for (let j = 0; j < H; j++) {
            const i = sig(gi[j] + gh[j]), f = sig(gi[H + j] + gh[H + j]);
            const g = Math.tanh(gi[2 * H + j] + gh[2 * H + j]), o = sig(gi[3 * H + j] + gh[3 * H + j]);
            c[j] = f * c[j] + i * g;
            h[j] = o * Math.tanh(c[j]);
          }
        } else {                             // GRU: compuertas r, z, n
          for (let j = 0; j < H; j++) {
            const r = sig(gi[j] + gh[j]), z = sig(gi[H + j] + gh[H + j]);
            const n = Math.tanh(gi[2 * H + j] + r * gh[2 * H + j]);
            h[j] = (1 - z) * n + z * h[j];
          }
        }
        salida.set(h, t * H);
      }
      entrada = salida;
      nIn = H;
    }
    const Hs = W["cabeza.1.bias"].length, y = new Float32Array(Hs);
    lineal(W["cabeza.1.weight"], W["cabeza.1.bias"], salida, (L - 1) * H, H, Hs, y, 0);
    return y;
  }

  // ---------- Transformer encoder (norm_first) ----------
  function normCapa(x, xo, g, b, d, y, yo) {
    let m = 0;
    for (let i = 0; i < d; i++) m += x[xo + i];
    m /= d;
    let v = 0;
    for (let i = 0; i < d; i++) { const t = x[xo + i] - m; v += t * t; }
    const inv = 1 / Math.sqrt(v / d + 1e-5);
    for (let i = 0; i < d; i++) y[yo + i] = (x[xo + i] - m) * inv * g[i] + b[i];
  }

  function transformer(red, x, L, F, p) {
    const W = red.W, d = W["entrada.bias"].length, nh = 4, dh = d / nh, ff = W["encoder.layers.0.linear1.bias"].length;
    const z = new Float32Array(L * d), a = new Float32Array(L * d), qkv = new Float32Array(L * 3 * d);
    const ctx = new Float32Array(L * d), tmp = new Float32Array(d), oculto = new Float32Array(ff), pesos = new Float32Array(L);
    for (let t = 0; t < L; t++) {
      lineal(W["entrada.weight"], W["entrada.bias"], x, t * F, F, d, z, t * d);
      for (let i = 0; i < d; i++) z[t * d + i] += W["pos"][t * d + i] + W["emb.weight"][p * d + i];
    }
    const esc = 1 / Math.sqrt(dh);
    for (let k = 0; k < 2; k++) {
      const pre = `encoder.layers.${k}.`;
      // bloque de autoatención
      for (let t = 0; t < L; t++) normCapa(z, t * d, W[pre + "norm1.weight"], W[pre + "norm1.bias"], d, a, t * d);
      for (let t = 0; t < L; t++) lineal(W[pre + "self_attn.in_proj_weight"], W[pre + "self_attn.in_proj_bias"], a, t * d, d, 3 * d, qkv, t * 3 * d);
      for (let hh = 0; hh < nh; hh++) {
        for (let t = 0; t < L; t++) {
          let mx = -Infinity;
          for (let s = 0; s < L; s++) {
            let sc = 0;
            for (let i = 0; i < dh; i++) sc += qkv[t * 3 * d + hh * dh + i] * qkv[s * 3 * d + d + hh * dh + i];
            pesos[s] = sc * esc;
            if (pesos[s] > mx) mx = pesos[s];
          }
          let suma = 0;
          for (let s = 0; s < L; s++) { pesos[s] = Math.exp(pesos[s] - mx); suma += pesos[s]; }
          for (let i = 0; i < dh; i++) {
            let acc = 0;
            for (let s = 0; s < L; s++) acc += pesos[s] * qkv[s * 3 * d + 2 * d + hh * dh + i];
            ctx[t * d + hh * dh + i] = acc / suma;
          }
        }
      }
      for (let t = 0; t < L; t++) {
        lineal(W[pre + "self_attn.out_proj.weight"], W[pre + "self_attn.out_proj.bias"], ctx, t * d, d, d, tmp, 0);
        for (let i = 0; i < d; i++) z[t * d + i] += tmp[i];
      }
      // bloque feed-forward
      for (let t = 0; t < L; t++) {
        normCapa(z, t * d, W[pre + "norm2.weight"], W[pre + "norm2.bias"], d, a, t * d);
        lineal(W[pre + "linear1.weight"], W[pre + "linear1.bias"], a, t * d, d, ff, oculto, 0);
        for (let i = 0; i < ff; i++) if (oculto[i] < 0) oculto[i] = 0;
        lineal(W[pre + "linear2.weight"], W[pre + "linear2.bias"], oculto, 0, ff, d, tmp, 0);
        for (let i = 0; i < d; i++) z[t * d + i] += tmp[i];
      }
    }
    normCapa(z, (L - 1) * d, W["norm.weight"], W["norm.bias"], d, tmp, 0);
    const Hs = W["cabeza.bias"].length, y = new Float32Array(Hs);
    lineal(W["cabeza.weight"], W["cabeza.bias"], tmp, 0, d, Hs, y, 0);
    return y;
  }

  /* Pronóstico de una red: crudo = filas (L x F) de variables sin estandarizar; devuelve log(1+casos) para H semanas */
  function pronosticarRed(red, crudo, L, F, p) {
    const x = new Float32Array(L * F);
    for (let t = 0; t < L; t++) for (let f = 0; f < F; f++) x[t * F + f] = (crudo[t * F + f] - red.media[f]) / red.desv[f];
    const sal = red.arq === "transformer" ? transformer(red, x, L, F, p) : recurrente(red, x, L, F, p);
    const ult = crudo[(L - 1) * F];           // la primera variable es log(1+casos)
    const out = new Float64Array(sal.length);
    for (let h = 0; h < sal.length; h++) out[h] = Math.max(0, sal[h] * red.yDesv + red.yMedia + (red.residual ? ult : 0));
    return out;
  }

  const api = {decodificar, cargarRed, pronosticarRed};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else raiz.Modelos = api;
})(typeof window !== "undefined" ? window : globalThis);
