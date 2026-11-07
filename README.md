# Pronóstico de dengue y EDA en la costa norte del Perú para planificar el abastecimiento de medicamentos

Proyecto integrador · MIA-203 Redes Neuronales y Aprendizaje Profundo · UNI FIIS · **Grupo 5**

Integrantes: Fernando Tapia Álvarez, Ana Rocio Cuchuirumi Merma, Edgard Antonio Quispe Arias,
Matías Samuel Ochoa Alejos y Carlos Cassano Vilca.

- Versión entregada: etiqueta [`v1.0-final`](https://github.com/Kysard/dengue-costa-norte/releases/tag/v1.0-final)
- Pesos entrenados: [`pesos_grupo5.zip`](https://github.com/Kysard/dengue-costa-norte/releases/download/v1.0-final/pesos_grupo5.zip) (también en `results/pesos/`)

La pregunta del proyecto es si se puede saber con varias semanas de anticipación cuántos casos de **dengue**
(y de **EDA en niños menores de 5 años**) va a tener cada provincia de Tumbes, Piura y Lambayeque, para mandar
los medicamentos a tiempo. Usamos los datos abiertos del CDC Perú (2000-2025), el clima de NASA POWER y los
índices de El Niño de la NOAA, y comparamos una LSTM, una GRU y un Transformer contra métodos clásicos. Con el
pronóstico simulamos el almacén de cada provincia con una política de stock base.

**Aplicativo:** abrir **`app/index.html`** en Chrome o Edge (doble clic, funciona sin internet) o usarlo en línea en
**<https://dengue-costa-norte.vercel.app>**. Las redes
entrenadas corren dentro del navegador: se elige provincia, modelo y de 1 a 26 semanas hacia adelante, se plantean
escenarios de clima y El Niño, se pueden corregir los casos de las últimas semanas y cada pronóstico se guarda para
compararlo semana a semana. Con «Avanzar en el tiempo» el pronóstico se agrega al historial como si ya hubiera pasado
y el modelo sigue pronosticando desde ahí (de una en una, de cuatro en cuatro o solo), y un botón compara los seis
escenarios de clima a la vez. También calcula el pedido de medicamentos de las 14 provincias.

**Artículo:** `paper/Equipo5_Paper.pdf` (formato IEEE, fuente LaTeX en la misma carpeta).

## Resultados principales

Conjunto de prueba 2021-2025; "MAE val." es el error en validación 2017-2020. Media ± d.e. de 5 semillas.
MAE y RMSE en casos por provincia y semana, promedio de los horizontes 1 a 8; F1 = anticipar con 4 semanas una
semana de alerta (sobre el percentil 90 de los 5 años previos).

**Dengue**

| Modelo | MAE val. | MAE | RMSE | MAE_log | F1 |
|---|---|---|---|---|---|
| Persistencia | 17.1 | 38.5 | 171.7 | 0.647 | 0.807 |
| Canal endémico (5 años) | 25.3 | 56.0 | 218.1 | 1.634 | 0.008 |
| ARIMA + Fourier | 12.7 | 34.5 | 161.6 | 0.646 | 0.757 |
| Random forest | 11.3 | 29.9 ± 0.2 | 149.3 ± 0.7 | 0.612 ± 0.001 | 0.803 ± 0.003 |
| LSTM | 11.1 ± 0.8 | 31.8 ± 0.8 | 157.0 ± 6.5 | 0.605 ± 0.011 | 0.809 ± 0.007 |
| GRU | 11.1 ± 0.5 | 31.7 ± 0.9 | 151.9 ± 4.9 | 0.619 ± 0.012 | 0.806 ± 0.007 |
| Transformer | 10.9 ± 0.5 | 29.8 ± 0.9 | 147.8 ± 5.0 | 0.607 ± 0.022 | 0.808 ± 0.008 |

**EDA en menores de 5 años**

| Modelo | MAE val. | MAE | RMSE | MAE_log | F1 |
|---|---|---|---|---|---|
| Persistencia | 15.8 | 14.3 | 27.9 | 0.338 | 0.496 |
| Canal endémico (5 años) | 21.7 | 20.9 | 38.3 | 0.460 | 0.014 |
| ARIMA + Fourier | 15.2 | 14.7 | 27.1 | 0.310 | 0.534 |
| Random forest | 14.2 | 13.6 | 27.0 | 0.305 | 0.509 ± 0.003 |
| LSTM | 15.5 ± 0.2 | 14.0 ± 0.3 | 26.0 ± 0.7 | 0.331 ± 0.005 | 0.489 ± 0.010 |
| GRU | 15.2 ± 0.1 | 13.5 ± 0.3 | 24.8 ± 0.8 | 0.322 ± 0.005 | 0.487 ± 0.012 |
| Transformer | 14.9 ± 0.4 | 13.8 ± 0.5 | 25.4 ± 0.5 | 0.318 ± 0.011 | 0.476 ± 0.032 |

- A **8 semanas de anticipación** el error de dengue baja de 59.5 casos (persistencia) a 40.2 con el Transformer
  (33 % menos) y a 42.3 con la LSTM. A 1 semana la persistencia y el random forest son igual de buenos que las redes.
- En promedio, el Transformer reduce el MAE de dengue 22.6 % frente a la persistencia y 13.6 % frente a ARIMA; el
  random forest queda prácticamente empatado. En EDA la mejora es menor (5.6 % con la GRU) y el random forest empata.
- **Ablación (LSTM, dengue):** sin la conexión residual el MAE sube de 31.8 a 40.4. Sin clima empeora la validación
  (11.1 → 11.7) y el RMSE (157.0 → 161.6), pero en prueba los modelos sin clima o sin El Niño quedan igual o un
  poco mejor (diferencias del orden de la variación entre semillas).
- **Gradientes integrados:** historia de casos 45 %, estacionalidad 30 %, clima 20 %, El Niño 5 %.
- **Abastecimiento (stock base, entrega en 4 semanas, z = 1.65):** con la LSTM se atiende el 71.9 % de la demanda
  frente al 44.9 % planificando con el canal endémico; los tratamientos no atendidos bajan 49 % y el inventario
  promedio 46 % (2,897 vs 5,340 tratamientos por semana).

Los índices de El Niño entran con el rezago con que la NOAA los publica (Niño 1+2 del mes anterior, ONI de la
estación centrada dos meses antes), para que el modelo no use información del futuro.

Las tablas completas están en `results/tablas/`, las figuras en `results/figuras/` y el registro de cada red entrenada (épocas, pérdida de validación y tiempo) en `results/logs/entrenamiento_redes.csv`.

## Estructura

```
dengue-costa-norte/
├── README.md
├── LICENSE, .gitignore
├── requirements.txt
├── configs/experimentos.yaml        # hiperparámetros de todos los experimentos
├── data/                            # ver data/README.md (fuentes, licencias y limpieza)
│   ├── raw/                         # datos agregados del CDC, NASA y NOAA
│   ├── external/                    # provincias, censos y kit de medicamentos
│   └── processed/panel_provincia_semana.csv
├── src/
│   ├── descargar_datos.py           # baja los datos originales
│   ├── construir_dataset.py         # arma el panel provincia-semana
│   ├── exportar_excel.py            # Excel del dataset con fórmulas y fuentes
│   ├── datos.py                     # variables, partición temporal y ventanas
│   ├── modelos.py                   # LSTM, GRU y Transformer (PyTorch)
│   ├── entrenar.py                  # entrenamiento con parada temprana y semillas
│   ├── lineas_base.py               # persistencia, canal endémico, ARIMA + Fourier, random forest
│   ├── evaluar.py                   # MAE, RMSE, F1 de alertas
│   ├── interpretar.py               # gradientes integrados
│   ├── abastecimiento.py            # simulación de stock base
│   ├── inferencia.py                # pronóstico desde la última semana + medicamentos
│   ├── graficos.py                  # figuras
│   ├── experimentos.py              # corre todo desde la terminal
│   ├── exportar_app.py              # arma el aplicativo web
│   └── articulo.py                  # figuras, tablas y números del artículo
├── notebooks/
│   ├── Proyecto_Grupo5_Dengue_EDA.ipynb   # notebook principal (ejecutado)
│   └── Demo_inferencia.ipynb              # carga los pesos y pronostica
├── results/
│   ├── tablas/  figuras/  pesos/  logs/
│   └── Dataset_Dengue_EDA_CostaNorte.xlsx
├── app/
│   ├── index.html                   # aplicativo para pronosticar (se abre con doble clic)
│   ├── modelo.js                    # LSTM, GRU y Transformer en JavaScript (mismos pesos)
│   ├── plantilla.html, mapa.json    # se usan para regenerarlo
│   └── lib/chart.umd.js             # Chart.js para que funcione sin internet
└── paper/
    ├── Equipo5_Paper.pdf, Equipo5_Paper.tex
    ├── referencias.bib, numeros.tex, texto_eda.tex
    └── figuras/, tablas/            # generadas por src/articulo.py
```

## Cómo reproducir

**En Google Colab (recomendado, con GPU T4):** abrir el notebook desde GitHub
(*Archivo > Abrir notebook > GitHub* y pegar `https://github.com/Kysard/dengue-costa-norte`), elegir `notebooks/Proyecto_Grupo5_Dengue_EDA.ipynb` y
*Ejecutar todo*. La primera celda clona el repositorio; si no hay conexión con GitHub, pide subir el zip del proyecto.
Para solo probar los pesos guardados, usar `notebooks/Demo_inferencia.ipynb` (tarda menos de un minuto).

**En una PC:**

```bash
git clone https://github.com/Kysard/dengue-costa-norte.git
cd dengue-costa-norte
pip install -r requirements.txt

# 1) Datos (opcional: data/ ya trae los archivos agregados)
python src/descargar_datos.py --data data
python src/construir_dataset.py --data data
python src/exportar_excel.py --data data --salida results/Dataset_Dengue_EDA_CostaNorte.xlsx

# 2) Experimentos: líneas base, LSTM/GRU/Transformer con 5 semillas, ablación y simulación
python src/experimentos.py --objetivo dengue
python src/experimentos.py --objetivo eda
#    (agregar --rapido para una sola semilla)

# 3) Todas las tablas y figuras del artículo
jupyter nbconvert --to notebook --execute notebooks/Proyecto_Grupo5_Dengue_EDA.ipynb --inplace

# 4) Demo con los pesos guardados
jupyter nbconvert --to notebook --execute notebooks/Demo_inferencia.ipynb --inplace

# 5) Aplicativo web (exporta los pesos guardados para que corran en el navegador)
python src/exportar_app.py

# 6) Figuras, tablas y números del artículo, y compilación
python src/articulo.py
cd paper && latexmk -pdf Equipo5_Paper.tex
```

- **Semillas:** 0, 1, 2, 3 y 4 (`configs/experimentos.yaml`).
- **Partición sin fuga:** entrenamiento 2000-2016, validación 2017-2020 (parada temprana), prueba 2021-2025.
  Una ventana se usa solo si sus 8 semanas objetivo caen en el mismo periodo; la estandarización usa solo el entrenamiento.
- **Pesos:** `results/pesos/{objetivo}_{modelo}_s{semilla}.pt` (30 archivos, 7 MB). También se pueden bajar en un solo
  zip desde el release [`v1.0-final`](https://github.com/Kysard/dengue-costa-norte/releases/download/v1.0-final/pesos_grupo5.zip); descomprimirlo dentro de `results/pesos/`.

## Despliegue en Vercel

El aplicativo es **estático**: `app/index.html` ya trae el JavaScript de las redes, los pesos y los datos
embebidos, así que no necesita backend ni Python en el servidor. El repositorio incluye un `vercel.json` que le
dice a Vercel que publique la carpeta `app` como sitio estático y que no intente detectar un entrypoint de Python
(por eso antes fallaba con `PYTHON_ENTRYPOINT_NOT_FOUND`).

1. En [vercel.com/new](https://vercel.com/new), importar el repositorio `Kysard/dengue-costa-norte`.
2. Framework Preset: **Other** (ya queda fijado en `vercel.json`). No hace falta Build Command, Output Directory
   ni variables de entorno.
3. Deploy. La aplicación queda en `https://<proyecto>.vercel.app` y se actualiza con cada push a `main`.

También se puede desplegar desde la terminal:

```bash
npm i -g vercel
vercel        # primera vez: vincula el proyecto
vercel --prod # despliegue a producción
```

`vercel.json` fuerza el preset «Other» y `outputDirectory: "app"`; `.vercelignore` deja fuera del despliegue
`src/`, `data/`, `results/`, `paper/` y `notebooks/`, que solo se usan para reproducir los experimentos.

## Hardware y costo

El notebook completo (líneas base, 3 redes × 5 semillas × 2 enfermedades, ablación, interpretabilidad y
simulación) corrió en **36 minutos en CPU** (Intel Xeon de 2 núcleos a 2.1 GHz, 8 GB de RAM, sin GPU); las 30
redes de la comparación principal suman unos 20 minutos. Cada red tiene entre 41 mil (GRU) y 71 mil
(Transformer) parámetros. En el aplicativo, las 15 redes de una enfermedad pronostican las 14 provincias en menos
de un segundo. Detalle en `results/tablas/costo_computo.csv`.

## Datos y licencias

CDC Perú (ODC-By), NASA POWER (uso libre citando al proyecto), NOAA CPC (dominio público), censos del INEI.
Detalle, huellas SHA-256 y limpieza en `data/README.md` y en la hoja `Fuentes` del Excel. Los datos son
agregados (sin información personal).

## Limitaciones

Datos de vigilancia con subnotificación; la clasificación de gravedad cambió en 2010; el clima se toma en la
capital de cada provincia; desde 2024-SE16 la fuente no reporta EDA en mayores de 5 años; las cantidades de
medicamentos por caso son supuestos del equipo; la simulación no considera vencimientos ni capacidad de almacén.

## Herramientas

Python, PyTorch, scikit-learn, statsmodels, pandas, matplotlib, Chart.js y Google Colab.
