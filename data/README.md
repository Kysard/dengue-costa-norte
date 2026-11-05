# Datos

Todo es público. Los archivos pesados originales no se suben al repositorio; aquí quedan los archivos ya
agregados (pocos MB) y el script para volver a generarlos.

| Carpeta / archivo | Contenido | Fuente y licencia |
|---|---|---|
| `raw/dengue_provincia_semana_peru_2000_2025.csv` | Casos de dengue por provincia, año, semana, gravedad (SSA, CSA, GRAVE) y tipo de diagnóstico (C, P), todo el Perú | CDC Perú – [Vigilancia epidemiológica de dengue](https://www.datosabiertos.gob.pe/dataset/vigilancia-epidemiol%C3%B3gica-de-dengue), ODC-By |
| `raw/eda_provincia_semana_costanorte_2000_2025.csv` | Episodios, hospitalizados y defunciones por EDA (< 5 y ≥ 5 años) en Tumbes, Piura y Lambayeque | CDC Perú – [Vigilancia epidemiológica de EDA](https://www.datosabiertos.gob.pe/dataset/vigilancia-epidemiol%C3%B3gica-de-enfermedad-diarreica-aguda-eda), ODC-By |
| `raw/eda_departamento_semana_peru_2000_2025.csv` | EDA por departamento y semana (todo el Perú, contexto) | ídem |
| `raw/clima_nasa_power_diario_costanorte_1999_2025.csv` | Temperatura (media, máx., mín.), lluvia y humedad diarias en la capital de cada provincia | [NASA POWER](https://power.larc.nasa.gov) Daily API v2.10.1, uso libre citando al proyecto |
| `raw/oni.ascii.txt` | Índice Oceánico El Niño (ONI) | [NOAA CPC](https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt), dominio público |
| `raw/nino12_cpc_1999_2026.txt` | Temperatura y anomalía mensual del Niño 1+2 (formato compacto `AAAAMM sst anom;`) | [NOAA CPC sstoi.indices](https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices), dominio público |
| `raw/manifest.json` | Tamaño y huella SHA-256 de los CSV originales del CDC usados | — |
| `external/provincias.csv` | 14 provincias, código INEI y coordenadas de su capital | INEI |
| `external/poblacion_censos.csv` | Población de los censos 1993, 2007 y 2017 | INEI (vía City Population) |
| `external/kit_medicamentos.csv` | Medicamentos por caso según grupo clínico (supuestos del equipo, editables) | Guía MINSA RM 071-2017, CDC, OMS/UNICEF |
| `processed/panel_provincia_semana.csv` | Panel final: 14 provincias × 1357 semanas = 18 998 filas, 29 columnas | generado por `src/construir_dataset.py` |

## Cómo regenerar todo desde cero

```bash
python src/descargar_datos.py --data data      # baja ~235 MB del CDC + NASA POWER + NOAA
python src/construir_dataset.py --data data    # arma data/processed/panel_provincia_semana.csv
python src/exportar_excel.py --data data       # Excel con el dataset, fórmulas y fuentes
```

## Limpieza aplicada

1. Registros con semana 53 en años que solo tienen 52 semanas se pasan a la semana 52 (9 casos de dengue de 2009 y 23 episodios de EDA de 2019 y 2023 en la costa norte).
2. La Ñ viene dañada en el archivo del CDC (`FERRE�AFE`); se corrige a `FERREÑAFE`.
3. Una fila del CSV nacional de dengue (Loreto, 2004) tiene comillas mal cerradas y se descarta; no afecta a la costa norte.
4. Semanas sin notificación = 0 casos.
5. La lluvia de la última semana de 2025 (solo 4 días con datos) se lleva a 7 días.
6. **Ojo:** desde la semana 16 de 2024 el archivo abierto de EDA ya no trae episodios en mayores de 5 años. El modelo de EDA usa solo menores de 5 años.
7. Antes de 2010 no existía la categoría "con signos de alarma" (clasificación OMS 2009). El modelo pronostica el total de casos.

Los archivos de `raw/` de dengue, EDA y clima se descargaron y agregaron el 27/10/2026 desde el navegador
(con la misma lógica que `src/descargar_datos.py`). Los totales se verificaron contra la fuente: 342 671 casos de
dengue y 2 688 585 episodios de EDA en la costa norte.
