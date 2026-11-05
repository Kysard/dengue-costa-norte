"""
Exporta el dataset a Excel (una hoja por tabla, con fórmulas y referencias).

Uso:
  python src/exportar_excel.py --data data --salida results/Dataset_Dengue_EDA_CostaNorte.xlsx
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

import construir_dataset as cd

FUENTE = "Arial"
F_NORMAL = Font(name=FUENTE, size=10)
F_NEGRITA = Font(name=FUENTE, size=10, bold=True)
F_TITULO = Font(name=FUENTE, size=14, bold=True)
F_SUBT = Font(name=FUENTE, size=11, bold=True)
F_ENTRADA = Font(name=FUENTE, size=10, color="0000FF")
F_LINK = Font(name=FUENTE, size=10, color="008000")
F_NOTA = Font(name=FUENTE, size=9, italic=True, color="555555")
F_URL = Font(name=FUENTE, size=10, color="0563C1", underline="single")
RELLENO_CAB = PatternFill("solid", fgColor="D9E1F2")
RELLENO_SUP = PatternFill("solid", fgColor="FFFF00")
BORDE = Border(bottom=Side(style="thin", color="999999"))
ACCESO = "27/10/2026"


def cabecera(ws, fila, textos, col_ini=1):
    for j, t in enumerate(textos):
        c = ws.cell(row=fila, column=col_ini + j, value=t)
        c.font = F_NEGRITA
        c.fill = RELLENO_CAB
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BORDE


def anchos(ws, valores):
    for j, w in enumerate(valores, start=1):
        ws.column_dimensions[get_column_letter(j)].width = w


def titulo(ws, texto, sub=None):
    ws["A1"] = texto
    ws["A1"].font = F_TITULO
    if sub:
        ws["A2"] = sub
        ws["A2"].font = F_NOTA


# ---------------------------------------------------------------------------
def hoja_panel(wb, panel, n_prov):
    ws = wb.create_sheet("Panel_semanal")
    cols = list(panel.columns)
    cabecera(ws, 1, cols)
    formulas = {"dengue_total": "=G{r}+H{r}+I{r}",
                "eda_total": "=M{r}+N{r}",
                "poblacion": "=INDEX(Poblacion!$G$5:$AF${u},MATCH(A{r},Poblacion!$A$5:$A${u},0),D{r}-1999)",
                "inc_dengue_100k": "=IF(AA{r}>0,J{r}/AA{r}*100000,0)",
                "inc_eda_100k": "=IF(AA{r}>0,O{r}/AA{r}*100000,0)"}
    u = 4 + n_prov
    for i, fila in enumerate(panel.itertuples(index=False), start=2):
        for j, (col, v) in enumerate(zip(cols, fila), start=1):
            if col in formulas:
                ws.cell(row=i, column=j, value=formulas[col].format(r=i, u=u))
            else:
                ws.cell(row=i, column=j, value=v)
    n = len(panel) + 1
    for col_letra, fmt in {"F": "dd/mm/yyyy", "S": "0.00", "T": "0.00", "U": "0.00", "V": "0.0",
                           "W": "0.0", "X": "0.00", "Y": "0.00", "Z": "0.00", "AA": "#,##0",
                           "AB": "0.00", "AC": "0.00"}.items():
        for r in range(2, n + 1):
            ws[f"{col_letra}{r}"].number_format = fmt
    ws.freeze_panes = "G2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{n}"
    anchos(ws, [9, 13, 22, 6, 7, 11] + [10] * 23)
    ws.row_dimensions[1].height = 30
    return ws


def hoja_poblacion(wb, pob, prov):
    ws = wb.create_sheet("Poblacion")
    titulo(ws, "Población por provincia (censos INEI y estimación anual)",
           "Censos 1993, 2007 y 2017 (dato de entrada). Años 2000-2025: interpolación geométrica entre censos; "
           "después de 2017 se extrapola con la tasa 2007-2017. Estimación propia del equipo.")
    cab = ["ubigeo_prov", "departamento", "provincia", "censo_1993", "censo_2007", "censo_2017"] + \
          [str(a) for a in range(2000, 2026)]
    cabecera(ws, 4, cab)
    m = pob.merge(prov[["ubigeo_prov", "departamento"]], on="ubigeo_prov")
    for i, r in enumerate(m.itertuples(index=False), start=5):
        ws.cell(row=i, column=1, value=r.ubigeo_prov)
        ws.cell(row=i, column=2, value=r.departamento)
        ws.cell(row=i, column=3, value=cd.corregir_texto(pd.Series([r.provincia]))[0])
        for j, v in enumerate([r.censo_1993, r.censo_2007, r.censo_2017], start=4):
            c = ws.cell(row=i, column=j, value=int(v))
            c.font = F_ENTRADA
            c.number_format = "#,##0"
        for k, a in enumerate(range(2000, 2026)):
            col = 7 + k
            letra = get_column_letter(col)
            if a < 2007:
                f = f"=ROUND($D{i}*($E{i}/$D{i})^(({letra}$4-1993)/14),0)"
            else:
                f = f"=ROUND($E{i}*($F{i}/$E{i})^(({letra}$4-2007)/10),0)"
            c = ws.cell(row=i, column=col, value=f)
            c.number_format = "#,##0"
    # años como número para las fórmulas (cabecera numérica pero mostrada como texto de año)
    for k, a in enumerate(range(2000, 2026)):
        c = ws.cell(row=4, column=7 + k, value=a)
        c.number_format = "0"
    fin = 4 + len(m)
    ws.cell(row=fin + 2, column=1,
            value="Fuente: INEI, Censos Nacionales 1993, 2007 y 2017 (cifras por provincia compiladas en "
                  "City Population). Ver hoja Fuentes [6].").font = F_NOTA
    ws.cell(row=fin + 3, column=1,
            value="Nota: la población censada no corrige la omisión censal; las estimaciones oficiales del INEI "
                  "pueden ser algo mayores.").font = F_NOTA
    for c in ws[4]:
        c.font = F_NEGRITA
    anchos(ws, [11, 13, 22, 11, 11, 11] + [9] * 26)
    ws.freeze_panes = "D5"
    return ws


def hoja_resumen(wb, prov, n):
    ws = wb.create_sheet("Resumen_anual")
    titulo(ws, "Resumen anual por provincia (calculado con fórmulas desde Panel_semanal)")
    anios = list(range(2000, 2026))
    bloques = [("Casos de dengue (confirmados + probables)", "J"),
               ("Episodios de EDA (todas las edades)", "O")]
    fila = 4
    for nombre, col in bloques:
        ws.cell(row=fila, column=1, value=nombre).font = F_SUBT
        fila += 1
        cabecera(ws, fila, ["ubigeo_prov", "departamento", "provincia"] + anios + ["Total"])
        for j in range(4, 4 + len(anios)):
            ws.cell(row=fila, column=j).number_format = "0"
        cab = fila
        for r in prov.itertuples(index=False):
            fila += 1
            ws.cell(row=fila, column=1, value=r.ubigeo_prov)
            ws.cell(row=fila, column=2, value=r.departamento)
            ws.cell(row=fila, column=3, value=r.provincia)
            for k, a in enumerate(anios):
                letra = get_column_letter(4 + k)
                c = ws.cell(row=fila, column=4 + k,
                            value=f"=SUMIFS(Panel_semanal!${col}$2:${col}${n},Panel_semanal!$A$2:$A${n},$A{fila},"
                                  f"Panel_semanal!$D$2:$D${n},{letra}${cab})")
                c.number_format = "#,##0"
            ult = get_column_letter(3 + len(anios))
            c = ws.cell(row=fila, column=4 + len(anios), value=f"=SUM(D{fila}:{ult}{fila})")
            c.number_format = "#,##0"
            c.font = F_NEGRITA
        fila += 1
        ws.cell(row=fila, column=3, value="Total costa norte").font = F_NEGRITA
        for k in range(len(anios) + 1):
            letra = get_column_letter(4 + k)
            c = ws.cell(row=fila, column=4 + k, value=f"=SUM({letra}{cab + 1}:{letra}{fila - 1})")
            c.number_format = "#,##0"
            c.font = F_NEGRITA
        fila += 3
    anchos(ws, [11, 13, 22] + [8] * 26 + [10])
    ws.freeze_panes = "D4"
    return ws


def hoja_nacional(wb, den_nac):
    ws = wb.create_sheet("Dengue_nacional")
    titulo(ws, "Dengue en todo el Perú (contexto y validación)",
           "Valores agregados del archivo original del CDC Perú (todos los departamentos). "
           "Casos confirmados + probables por semana de inicio de síntomas.")
    anual = den_nac.pivot_table(index="departamento", columns="ano", values="casos", aggfunc="sum", fill_value=0)
    anual = anual.loc[anual.sum(axis=1).sort_values(ascending=False).index]
    ws["A4"] = "Casos por departamento y año"
    ws["A4"].font = F_SUBT
    cabecera(ws, 5, ["departamento"] + list(anual.columns) + ["Total"])
    for j in range(2, 2 + len(anual.columns)):
        ws.cell(row=5, column=j).number_format = "0"
    for i, (dep, fila) in enumerate(anual.iterrows(), start=6):
        ws.cell(row=i, column=1, value=cd.corregir_texto(pd.Series([dep]))[0])
        for j, v in enumerate(fila.values, start=2):
            ws.cell(row=i, column=j, value=int(v)).number_format = "#,##0"
        ult = get_column_letter(1 + len(anual.columns))
        c = ws.cell(row=i, column=2 + len(anual.columns), value=f"=SUM(B{i}:{ult}{i})")
        c.number_format = "#,##0"
    fin = 5 + len(anual)
    fila_tot = fin + 1
    ws.cell(row=fila_tot, column=1, value="Total Perú").font = F_NEGRITA
    for j in range(2, 3 + len(anual.columns)):
        letra = get_column_letter(j)
        c = ws.cell(row=fila_tot, column=j, value=f"=SUM({letra}6:{letra}{fin})")
        c.number_format = "#,##0"
        c.font = F_NEGRITA

    # serie semanal nacional (para validar contra la alerta del CDC 2023)
    col0 = 3 + len(anual.columns) + 2
    ws.cell(row=4, column=col0, value="Serie semanal nacional").font = F_SUBT
    cabecera(ws, 5, ["ano", "semana", "casos"], col_ini=col0)
    sem = den_nac.groupby(["ano", "semana"]).casos.sum().reset_index()
    for i, r in enumerate(sem.itertuples(index=False), start=6):
        ws.cell(row=i, column=col0, value=int(r.ano))
        ws.cell(row=i, column=col0 + 1, value=int(r.semana))
        ws.cell(row=i, column=col0 + 2, value=int(r.casos)).number_format = "#,##0"
    anchos(ws, [16] + [8] * 26 + [10])
    for k in range(3):
        ws.column_dimensions[get_column_letter(col0 + k)].width = 9
    ws.freeze_panes = "B6"
    return ws, col0, 5 + len(sem)


def hoja_enso(wb, enso):
    ws = wb.create_sheet("ENSO_mensual")
    titulo(ws, "Índices de El Niño (mensuales)",
           "Niño 1+2: temperatura del mar frente a la costa norte del Perú. ONI: anomalía del Niño 3.4 "
           "(promedio móvil de 3 meses, asignado al mes central). Fuente: NOAA CPC.")
    cabecera(ws, 4, ["ano", "mes", "nino12_sst (°C)", "nino12_anom (°C)", "oni (°C)"])
    for i, r in enumerate(enso.itertuples(index=False), start=5):
        ws.cell(row=i, column=1, value=int(r.ano_m))
        ws.cell(row=i, column=2, value=int(r.mes))
        for j, v in enumerate([r.nino12_sst, r.nino12_anom, r.oni], start=3):
            c = ws.cell(row=i, column=j, value=None if pd.isna(v) else float(v))
            c.number_format = "0.00"
    anchos(ws, [8, 6, 16, 17, 10])
    ws.freeze_panes = "A5"
    return ws


def hoja_provincias(wb, prov):
    ws = wb.create_sheet("Provincias")
    titulo(ws, "Provincias del estudio y punto usado para el clima",
           "El clima de NASA POWER se tomó en las coordenadas de la capital de cada provincia "
           "(resolución de la fuente: 0.5° x 0.625°).")
    cabecera(ws, 4, ["ubigeo_prov", "departamento", "provincia", "capital", "latitud", "longitud"])
    for i, r in enumerate(prov.itertuples(index=False), start=5):
        for j, v in enumerate([r.ubigeo_prov, r.departamento, r.provincia, r.capital, r.lat, r.lon], start=1):
            c = ws.cell(row=i, column=j, value=v)
            if j >= 5:
                c.font = F_ENTRADA
                c.number_format = "0.000"
    anchos(ws, [11, 13, 22, 18, 10, 10])
    return ws


def hoja_kit(wb, kit):
    ws = wb.create_sheet("Kit_medicamentos")
    titulo(ws, "Supuestos de consumo de medicamentos por caso (editable)",
           "Las celdas amarillas son supuestos del equipo para la simulación de abastecimiento; deben validarse "
           "con la guía clínica del MINSA (RM 071-2017/MINSA) antes del artículo.")
    cab = ["enfermedad", "grupo_clinico", "variable_panel", "medicamento", "unidad",
           "cantidad_por_caso", "base_de_la_cantidad"]
    cabecera(ws, 3, cab)
    for i, r in enumerate(kit.itertuples(index=False), start=4):
        for j, v in enumerate(r, start=1):
            c = ws.cell(row=i, column=j, value=v)
            if j == 6:
                c.font = F_ENTRADA
                c.fill = RELLENO_SUP
    fin = 3 + len(kit)
    ws.cell(row=fin + 2, column=1, value="Leyenda:").font = F_NEGRITA
    ws.cell(row=fin + 3, column=1, value="Amarillo / azul = supuesto editable; al cambiarlo se recalcula la hoja "
                                         "Demanda_medicamentos.").font = F_NOTA
    ws.cell(row=fin + 4, column=1, value="Qué medicamentos usa cada grupo: guía MINSA [7], CDC EE.UU. [8] y OMS/UNICEF "
                                         "[10]. Lo que el MINSA distribuye en una epidemia real: Andina 2024 [9].").font = F_NOTA
    anchos(ws, [10, 40, 15, 28, 13, 12, 70])
    ws.row_dimensions[3].height = 30
    return ws, 4, fin


def hoja_demanda(wb, n, k_ini, k_fin):
    ws = wb.create_sheet("Demanda_medicamentos")
    titulo(ws, "Demanda histórica de medicamentos estimada a partir de los casos")
    ws["A3"] = "Departamento (elegir):"
    ws["A3"].font = F_NEGRITA
    ws["C3"] = "*"
    ws["C3"].font = F_ENTRADA
    ws["C3"].fill = RELLENO_SUP
    ws["D3"] = "* = toda la costa norte; o elegir TUMBES, PIURA o LAMBAYEQUE"
    ws["D3"].font = F_NOTA
    dv = DataValidation(type="list", formula1='"*,TUMBES,PIURA,LAMBAYEQUE"', allow_blank=False)
    ws.add_data_validation(dv)
    dv.add("C3")

    casos = [("Dengue SSA", "G"), ("Dengue CSA", "H"), ("Dengue grave", "I"),
             ("EDA <5 años", "M"), ("EDA ≥5 años", "N"), ("EDA hospitalizados", None)]
    variables = ["dengue_ssa", "dengue_csa", "dengue_grave", "eda_men5", "eda_may5", "eda_hosp"]
    meds = ["Paracetamol 500 mg", "Sales de rehidratacion oral", "Cloruro de sodio 0.9% 1 L",
            "Equipo de venoclisis", "Sulfato de zinc 20 mg"]
    ws.cell(row=5, column=2, value="Casos").font = F_SUBT
    ws.cell(row=5, column=9, value="Unidades requeridas (casos x kit por caso)").font = F_SUBT
    cabecera(ws, 6, ["Año"] + [c[0] for c in casos] + [""] + meds)
    ws.cell(row=6, column=8).fill = PatternFill(fill_type=None)
    rng = lambda c: f"Panel_semanal!${c}$2:${c}${n}"
    crit = f'{rng("B")},$C$3,{rng("D")},$A{{r}}'
    kf = lambda col: f"Kit_medicamentos!${col}${k_ini}:${col}${k_fin}"
    for i, a in enumerate(range(2000, 2026), start=7):
        ws.cell(row=i, column=1, value=a)
        for j, (_, col) in enumerate(casos, start=2):
            if col:
                f = f"=SUMIFS({rng(col)},{crit.format(r=i)})"
            else:
                f = f"=SUMIFS({rng('P')},{crit.format(r=i)})+SUMIFS({rng('Q')},{crit.format(r=i)})"
            ws.cell(row=i, column=j, value=f).number_format = "#,##0"
        for m in range(len(meds)):
            col = 9 + m
            letra_med = get_column_letter(col)
            partes = []
            for v_idx, var in enumerate(variables):
                letra_caso = get_column_letter(2 + v_idx)
                partes.append(f'${letra_caso}{i}*SUMIFS({kf("F")},{kf("C")},"{var}",{kf("D")},{letra_med}$6)')
            ws.cell(row=i, column=col, value="=" + "+".join(partes)).number_format = "#,##0"
    fin = 6 + 26
    tot = fin + 1
    ws.cell(row=tot, column=1, value="Total").font = F_NEGRITA
    for col in list(range(2, 8)) + list(range(9, 14)):
        letra = get_column_letter(col)
        c = ws.cell(row=tot, column=col, value=f"=SUM({letra}7:{letra}{fin})")
        c.number_format = "#,##0"
        c.font = F_NEGRITA
    ws.cell(row=tot + 2, column=1, value="Las cantidades por caso vienen de la hoja Kit_medicamentos. "
                                          "Esta tabla muestra lo que se habría necesitado cada año; el pronóstico "
                                          "semanal del modelo se usa en el notebook para simular el stock.").font = F_NOTA
    anchos(ws, [8, 11, 11, 12, 11, 11, 13, 3, 13, 15, 15, 13, 13])
    ws.row_dimensions[6].height = 42
    ws.freeze_panes = "B7"
    return ws


def hoja_validacion(wb, n, col0_nac, fin_nac, panel):
    ws = wb.create_sheet("Validacion")
    titulo(ws, "Validación del dataset", "Comparación con cifras oficiales publicadas y controles de calidad.")
    cabecera(ws, 4, ["Indicador", "Valor oficial publicado", "Valor en el dataset", "Diferencia %", "Fuente",
                     "Comentario"])
    nac_a = get_column_letter(col0_nac)
    nac_s = get_column_letter(col0_nac + 1)
    nac_c = get_column_letter(col0_nac + 2)
    filas = [
        ("Dengue, región Tumbes, 2024, SE 1-48", 5664,
         f'=SUMIFS(Panel_semanal!$J$2:$J${n},Panel_semanal!$B$2:$B${n},"TUMBES",Panel_semanal!$D$2:$D${n},2024,'
         f'Panel_semanal!$E$2:$E${n},"<=48")',
         "[11] Boletín DIRESA Tumbes SE 48-2024 (2937 mujeres + 2727 hombres)",
         "El boletín cuenta confirmados y probables notificados a esa fecha; el dataset tiene la "
         "clasificación final, sin los probables que luego se descartaron."),
        ("Dengue, todo el Perú, 2023, SE 1-31", 230209,
         f"=SUMIFS(Dengue_nacional!${nac_c}$6:${nac_c}${fin_nac},Dengue_nacional!${nac_a}$6:${nac_a}${fin_nac},2023,"
         f'Dengue_nacional!${nac_s}$6:${nac_s}${fin_nac},"<=31")',
         "[12] Alerta epidemiológica CDC Perú, 22/08/2023",
         "Misma razón: la alerta se emitió con datos preliminares en plena epidemia."),
    ]
    for i, (ind, ofi, f, fuente, com) in enumerate(filas, start=5):
        ws.cell(row=i, column=1, value=ind)
        c = ws.cell(row=i, column=2, value=ofi)
        c.font = F_ENTRADA
        c.number_format = "#,##0"
        ws.cell(row=i, column=3, value=f).number_format = "#,##0"
        ws.cell(row=i, column=4, value=f"=IF(B{i}>0,(C{i}-B{i})/B{i},0)").number_format = "0.0%"
        ws.cell(row=i, column=5, value=fuente)
        ws.cell(row=i, column=6, value=com)

    ws.cell(row=9, column=1, value="Controles de calidad").font = F_SUBT
    cabecera(ws, 10, ["Control", "Esperado", "Resultado", "¿OK?", "Cómo se calcula", ""])
    n_sem = sum(cd.semanas_del_anio(a) for a in range(2000, 2026))
    controles = [
        ("Filas del panel (14 provincias x semanas 2000-2025)", f"=14*{n_sem}", f"=COUNTA(Panel_semanal!A2:A{n})",
         f"14 provincias x {n_sem} semanas epidemiológicas"),
        ("Casos de dengue de la costa norte en el archivo fuente", int(panel.dengue_total.sum()),
         f"=SUM(Panel_semanal!J2:J{n})", "Esperado: suma de los 14 códigos de provincia en el archivo agregado del CDC"),
        ("Episodios de EDA de la costa norte en el archivo fuente", int(panel.eda_total.sum()),
         f"=SUM(Panel_semanal!O2:O{n})", "Esperado: suma del archivo agregado de EDA"),
        ("Filas donde confirmados + probables ≠ total", 0,
         f"=SUMPRODUCT(--(Panel_semanal!K2:K{n}+Panel_semanal!L2:L{n}<>Panel_semanal!J2:J{n}))",
         "Debe ser 0"),
        ("Celdas de clima o El Niño vacías", 0, f"=COUNTBLANK(Panel_semanal!S2:Z{n})", "Debe ser 0"),
        ("Semanas con lluvia negativa", 0, f'=COUNTIF(Panel_semanal!V2:V{n},"<0")', "Debe ser 0"),
    ]
    for i, (ctrl, esp, res, como) in enumerate(controles, start=11):
        ws.cell(row=i, column=1, value=ctrl)
        c = ws.cell(row=i, column=2, value=esp)
        c.number_format = "#,##0"
        if not str(esp).startswith("="):
            c.font = F_ENTRADA
        ws.cell(row=i, column=3, value=res).number_format = "#,##0"
        ws.cell(row=i, column=4, value=f'=IF(B{i}=C{i},"OK","REVISAR")')
        ws.cell(row=i, column=5, value=como)

    ws.cell(row=19, column=1, value="Correcciones aplicadas a los datos de origen").font = F_SUBT
    notas = [
        "1. Dengue 2009: 9 casos de la costa norte venían con semana 53 (2009 solo tiene 52); se pasaron a la semana 52.",
        "2. EDA 2019 y 2023: 23 episodios venían con semana 53 en años de 52 semanas; se pasaron a la semana 52.",
        "3. La letra Ñ viene dañada en el archivo del CDC (FERRE?AFE); se corrigió a FERREÑAFE.",
        "4. Una fila del CSV nacional de dengue (Loreto, 2004) tiene comillas mal cerradas y no se pudo leer; "
        "no afecta a la costa norte.",
        "5. Semanas sin notificación se registran como 0 casos.",
        "6. La última semana de 2025 (SE 53) solo tiene 4 días de clima; la lluvia se llevó a 7 días.",
        "7. Antes de 2010 no existía la categoría 'con signos de alarma' (clasificación OMS 2009): esos años solo "
        "tienen casos sin signos de alarma o graves. Para el modelo conviene pronosticar el total.",
        "8. En 2020-2021 la notificación de EDA cae por la pandemia de COVID-19 (menos consultas), no por menos "
        "enfermedad. Hay que tenerlo en cuenta al entrenar y al interpretar errores.",
        "9. IMPORTANTE: desde la semana 16 de 2024 el archivo abierto de EDA ya no trae episodios en mayores de "
        "5 años (vienen en 0 en todo el país). Por eso el modelo de EDA usa solo menores de 5 años (eda_men5), "
        "que sí es una serie completa 2000-2025.",
    ]
    for k, t in enumerate(notas, start=20):
        ws.cell(row=k, column=1, value=t)
    for r in range(5, 7):
        for col in (1, 5, 6):
            ws.cell(row=r, column=col).alignment = Alignment(wrap_text=True, vertical="top")
    anchos(ws, [52, 16, 16, 12, 45, 70])
    return ws


def hoja_fuentes(wb, manifest):
    ws = wb.create_sheet("Fuentes")
    titulo(ws, "Fuentes y referencias", f"Fecha de acceso de todas las fuentes: {ACCESO}.")
    cabecera(ws, 4, ["N°", "Referencia", "Uso en el dataset", "Enlace", "Licencia / condiciones", "Detalle técnico"])
    den = manifest["fuentes"]["dengue"]
    eda = manifest["fuentes"]["eda"]
    refs = [
        ("Centro Nacional de Epidemiología, Prevención y Control de Enfermedades [CDC Perú], MINSA. (2026). "
         "Vigilancia epidemiológica de dengue [Conjunto de datos]. Plataforma Nacional de Datos Abiertos.",
         "Casos de dengue 2000-2025 por provincia, semana, gravedad y tipo de diagnóstico",
         den["dataset"], "Open Data Commons Attribution (ODC-By)",
         f"Archivo datos_abiertos_vigilancia_dengue_2000_2025_0.csv; {den['bytes']:,} bytes; "
         f"modificado {den['fecha_modificada_portal']}; SHA-256 {den['sha256']}"),
        ("CDC Perú, MINSA. (2026). Vigilancia epidemiológica de enfermedad diarreica aguda (EDA) "
         "[Conjunto de datos]. Plataforma Nacional de Datos Abiertos.",
         "Episodios de EDA 2000-2025 en menores y mayores de 5 años",
         eda["dataset"], "Open Data Commons Attribution (ODC-By)",
         f"Archivo datos_abiertos_vigilancia_edas_2000_2025.csv; {eda['bytes']:,} bytes; "
         f"modificado {eda['fecha_modificada_portal']}; SHA-256 {eda['sha256']}"),
        ("NASA Langley Research Center. (2026). POWER Project, Daily API v2.10.1 (fuente MERRA-2).",
         "Temperatura, lluvia y humedad diarias 1999-2025 en las 14 capitales provinciales",
         "https://power.larc.nasa.gov", "Uso libre; se pide citar al proyecto POWER",
         "Parámetros T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, RH2M; comunidad AG"),
        ("NOAA Climate Prediction Center. (2026). Oceanic Niño Index (ONI).",
         "Índice ONI mensual", "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
         "Dominio público (gobierno de EE.UU.)", "ERSST v5; anomalía del Niño 3.4"),
        ("NOAA Climate Prediction Center. (2026). Monthly sea surface temperature indices (Niño 1+2).",
         "Temperatura y anomalía del mar en la región Niño 1+2", "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices",
         "Dominio público (gobierno de EE.UU.)", "Columnas NINO1+2 y ANOM, 1999-2026"),
        ("Instituto Nacional de Estadística e Informática [INEI]. Censos Nacionales 1993, 2007 y 2017 "
         "(cifras por provincia compiladas por T. Brinkhoff, City Population).",
         "Población por provincia para calcular la incidencia",
         "https://www.citypopulation.de/en/peru/admin/20__piura/", "Datos oficiales del INEI",
         "También: .../24__tumbes/ y .../14__lambayeque/"),
        ("Ministerio de Salud. (2017). Resolución Ministerial N.° 071-2017/MINSA: Guía de práctica clínica para la "
         "atención de casos de dengue en el Perú.",
         "Qué tratamiento recibe cada grupo clínico de dengue",
         "https://sni.org.pe/aprueban-la-guia-tecnica-guia-de-practica-clinica-para-la-atencion-de-casos-de-dengue-en-el-peru/",
         "Norma pública", "Revisar el texto oficial en gob.pe antes del artículo"),
        ("Centers for Disease Control and Prevention. (s. f.). Manejo clínico del dengue.",
         "Grupos A, B y C y su tratamiento (paracetamol, rehidratación oral, cristaloides EV)",
         "https://www.cdc.gov/dengue/es/healthcare-providers/treatment.html", "Dominio público", ""),
        ("Agencia Andina. (9 de marzo de 2024). Dengue: Minsa garantiza abastecimiento de medicamentos para "
         "pacientes en Piura.",
         "Referencia real de insumos que el MINSA distribuye en una epidemia",
         "https://andina.pe/ingles/noticia-dengue-minsa-garantiza-abastecimiento-medicamentos-para-pacientes-piura-977438.aspx",
         "Nota de prensa", "669 000 tabletas de paracetamol, 131 000 sueros y 289 000 sales de rehidratación"),
        ("World Health Organization & UNICEF. (2004). Clinical management of acute diarrhoea: WHO/UNICEF joint statement.",
         "Tratamiento de EDA (sales de rehidratación oral y zinc por 10-14 días)",
         "https://www.who.int/publications/i/item/WHO_FCH_CAH_04.7", "OMS", ""),
        ("Dirección Regional de Salud Tumbes. (2024). Boletín epidemiológico, semana 48-2024.",
         "Validación (casos de dengue Tumbes 2024)",
         "https://cdn.www.gob.pe/uploads/document/file/7391483/6303111-boletin_2024_tumbes-s48.pdf", "Documento público", ""),
        ("CDC Perú. (22 de agosto de 2023). CDC Perú emite alerta epidemiológica sobre epidemia de dengue.",
         "Validación (casos nacionales 2023 a la SE 31)", "https://www.dge.gob.pe/portalnuevo/?p=4861",
         "Documento público", ""),
    ]
    for i, (ref, uso, url, lic, det) in enumerate(refs, start=5):
        ws.cell(row=i, column=1, value=f"[{i - 4}]")
        ws.cell(row=i, column=2, value=ref)
        ws.cell(row=i, column=3, value=uso)
        c = ws.cell(row=i, column=4, value=url)
        c.hyperlink = url
        c.font = F_URL
        ws.cell(row=i, column=5, value=lic)
        ws.cell(row=i, column=6, value=det)
        for j in range(1, 7):
            ws.cell(row=i, column=j).alignment = Alignment(wrap_text=True, vertical="top")
    anchos(ws, [5, 60, 38, 45, 26, 50])
    return ws


def hoja_diccionario(wb):
    ws = wb.create_sheet("Diccionario", 1)
    titulo(ws, "Diccionario de variables de Panel_semanal")
    cabecera(ws, 3, ["Columna", "Variable", "Descripción", "Unidad", "Fuente", "Cálculo"])
    filas = [
        ("A", "ubigeo_prov", "Código INEI de la provincia (4 dígitos)", "código", "[1]", ""),
        ("B", "departamento", "Departamento", "texto", "[1]", ""),
        ("C", "provincia", "Provincia del lugar probable de infección", "texto", "[1]", ""),
        ("D", "ano", "Año epidemiológico", "año", "", ""),
        ("E", "semana", "Semana epidemiológica (domingo a sábado)", "1-53", "", ""),
        ("F", "fecha_inicio", "Domingo en que empieza la semana", "fecha", "", "Calendario epidemiológico"),
        ("G", "dengue_ssa", "Casos de dengue sin signos de alarma", "casos", "[1]", "Suma por semana de inicio de síntomas"),
        ("H", "dengue_csa", "Casos de dengue con signos de alarma", "casos", "[1]", ""),
        ("I", "dengue_grave", "Casos de dengue grave", "casos", "[1]", ""),
        ("J", "dengue_total", "Total de casos de dengue", "casos", "", "G + H + I"),
        ("K", "dengue_confirmados", "Casos confirmados (tipo_dx = C)", "casos", "[1]", ""),
        ("L", "dengue_probables", "Casos probables (tipo_dx = P)", "casos", "[1]", ""),
        ("M", "eda_men5", "Episodios de EDA en menores de 5 años", "episodios", "[2]", ""),
        ("N", "eda_may5", "Episodios de EDA en mayores de 5 años", "episodios", "[2]", "La fuente lo deja de reportar desde 2024-SE16"),
        ("O", "eda_total", "Total de episodios de EDA (incompleto desde 2024-SE16)", "episodios", "", "M + N"),
        ("P", "eda_hosp_men5", "Hospitalizados por EDA, menores de 5 años", "personas", "[2]", ""),
        ("Q", "eda_hosp_may5", "Hospitalizados por EDA, mayores de 5 años", "personas", "[2]", ""),
        ("R", "eda_defunciones", "Defunciones por EDA (todas las edades)", "personas", "[2]", ""),
        ("S", "t2m_media", "Temperatura media del aire a 2 m", "°C", "[3]", "Promedio de 7 días"),
        ("T", "t2m_max", "Temperatura máxima diaria (promedio de la semana)", "°C", "[3]", "Promedio de 7 días"),
        ("U", "t2m_min", "Temperatura mínima diaria (promedio de la semana)", "°C", "[3]", "Promedio de 7 días"),
        ("V", "precip_mm", "Lluvia acumulada en la semana", "mm", "[3]", "Suma de 7 días"),
        ("W", "hr_media", "Humedad relativa media a 2 m", "%", "[3]", "Promedio de 7 días"),
        ("X", "nino12_anom", "Anomalía de temperatura del mar, región Niño 1+2", "°C", "[5]", "Valor del mes de la semana"),
        ("Y", "nino12_sst", "Temperatura del mar, región Niño 1+2", "°C", "[5]", "Valor del mes de la semana"),
        ("Z", "oni", "Índice Oceánico El Niño (Niño 3.4)", "°C", "[4]", "Valor del mes de la semana"),
        ("AA", "poblacion", "Población estimada de la provincia en el año", "habitantes", "[6]",
         "Busca el valor en la hoja Poblacion"),
        ("AB", "inc_dengue_100k", "Incidencia semanal de dengue", "casos por 100 mil hab.", "", "J / AA x 100 000"),
        ("AC", "inc_eda_100k", "Incidencia semanal de EDA", "episodios por 100 mil hab.", "", "O / AA x 100 000"),
    ]
    for i, f in enumerate(filas, start=4):
        for j, v in enumerate(f, start=1):
            ws.cell(row=i, column=j, value=v)
    anchos(ws, [8, 18, 50, 22, 8, 32])
    return ws


def hoja_leeme(wb, panel):
    ws = wb.active
    ws.title = "Leeme"
    ws["A1"] = "Dataset: dengue, EDA y clima en la costa norte del Perú (2000-2025)"
    ws["A1"].font = F_TITULO
    ws["A2"] = "Proyecto integrador - MIA-203 Redes Neuronales y Aprendizaje Profundo (UNI FIIS) - Grupo 5"
    ws["A2"].font = F_NOTA
    lineas = [
        ("Objetivo", "Pronosticar los casos semanales de dengue y EDA por provincia a partir de datos históricos, "
                     "clima y El Niño, y usar ese pronóstico para planificar el abastecimiento de medicamentos."),
        ("Unidad de análisis", "Provincia x semana epidemiológica"),
        ("Cobertura", f"14 provincias de Tumbes, Piura y Lambayeque; 2000-SE01 a 2025-SE53; {len(panel):,} filas"),
        ("Casos incluidos", f"{int(panel.dengue_total.sum()):,} casos de dengue y {int(panel.eda_total.sum()):,} "
                            "episodios de EDA"),
        ("Fuentes", "CDC Perú (datos abiertos), NASA POWER, NOAA CPC, INEI. Detalle en la hoja Fuentes."),
        ("Fecha de descarga", ACCESO),
        ("Ojo", "Desde 2024-SE16 la fuente de EDA no reporta mayores de 5 años; por eso el modelo de EDA usa "
                "solo menores de 5 años. Ver la hoja Validacion."),
    ]
    for i, (k, v) in enumerate(lineas, start=4):
        ws.cell(row=i, column=1, value=k).font = F_NEGRITA
        ws.cell(row=i, column=2, value=v)
    ws.cell(row=12, column=1, value="Hojas del archivo").font = F_SUBT
    hojas = [
        ("Diccionario", "Qué significa cada columna del panel"),
        ("Panel_semanal", "Dataset principal para el modelo (una fila por provincia y semana)"),
        ("Resumen_anual", "Casos de dengue y EDA por provincia y año (fórmulas)"),
        ("Dengue_nacional", "Dengue en todos los departamentos del Perú (contexto)"),
        ("ENSO_mensual", "Índices de El Niño Niño 1+2 y ONI"),
        ("Poblacion", "Censos y población estimada por año"),
        ("Provincias", "Lista de provincias y coordenadas usadas para el clima"),
        ("Kit_medicamentos", "Supuestos de medicamentos por caso (editables)"),
        ("Demanda_medicamentos", "Unidades de medicamentos que se habrían necesitado cada año"),
        ("Validacion", "Comparación con cifras oficiales y controles de calidad"),
        ("Fuentes", "Referencias con enlace, licencia y huella del archivo"),
    ]
    for i, (h, d) in enumerate(hojas, start=13):
        c = ws.cell(row=i, column=1, value=h)
        c.hyperlink = f"#'{h}'!A1"
        c.font = F_URL
        ws.cell(row=i, column=2, value=d)
    ws.cell(row=25, column=1, value="Colores").font = F_SUBT
    ws.cell(row=26, column=1, value="Texto azul").font = F_ENTRADA
    ws.cell(row=26, column=2, value="Dato de entrada escrito a mano (censos, coordenadas, cifras oficiales)")
    ws.cell(row=27, column=1, value="Texto negro")
    ws.cell(row=27, column=2, value="Dato de la fuente o fórmula")
    c = ws.cell(row=28, column=1, value="Fondo amarillo")
    c.fill = RELLENO_SUP
    ws.cell(row=28, column=2, value="Supuesto que el equipo puede cambiar")
    ws.cell(row=30, column=1, value="Cómo se armó").font = F_SUBT
    pasos = [
        "1. Se descargaron los CSV de dengue y EDA del CDC Perú (Plataforma Nacional de Datos Abiertos).",
        "2. Se sumaron los casos por provincia, año, semana, gravedad y tipo de diagnóstico.",
        "3. Se descargó el clima diario de NASA POWER para la capital de cada provincia y se pasó a semanas.",
        "4. Se unieron los índices mensuales de El Niño (NOAA) según el mes de cada semana.",
        "5. Se calculó la población anual con los censos del INEI y la incidencia por 100 mil habitantes.",
        "6. Todo se puede regenerar con src/construir_dataset.py y src/exportar_excel.py.",
    ]
    for k, t in enumerate(pasos, start=31):
        ws.cell(row=k, column=1, value=t)
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 100
    return ws


def aplicar_fuente(wb):
    for ws in wb.worksheets:
        for fila in ws.iter_rows():
            for c in fila:
                if c.value is not None and (c.font is None or c.font.name != FUENTE):
                    c.font = F_NORMAL
                elif c.value is not None and c.font.name == FUENTE:
                    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--salida", default="results/Dataset_Dengue_EDA_CostaNorte.xlsx")
    args = ap.parse_args()
    data = Path(args.data)

    panel = pd.read_csv(data / "processed" / "panel_provincia_semana.csv", dtype={"ubigeo_prov": str},
                        parse_dates=["fecha_inicio"])
    panel["fecha_inicio"] = panel["fecha_inicio"].dt.date
    prov = pd.read_csv(data / "external" / "provincias.csv", dtype={"ubigeo_prov": str})
    pob = pd.read_csv(data / "external" / "poblacion_censos.csv", dtype={"ubigeo_prov": str})
    kit = pd.read_csv(data / "external" / "kit_medicamentos.csv")
    manifest = json.loads((data / "raw" / "manifest.json").read_text())
    den_nac = cd.corregir_semana_53(pd.read_csv(data / "raw" / "dengue_provincia_semana_peru_2000_2025.csv",
                                                dtype={"ubigeo_prov": str}))
    enso = cd.leer_nino12(data).merge(cd.leer_oni(data / "raw" / "oni.ascii.txt"), on=["ano_m", "mes"], how="left")
    enso = enso[enso.ano_m >= 1999]

    # el orden de las filas de Poblacion debe coincidir con provincias.csv
    pob = prov[["ubigeo_prov"]].merge(pob, on="ubigeo_prov")

    wb = Workbook()
    hoja_leeme(wb, panel)
    n = len(panel) + 1
    hoja_panel(wb, panel, len(prov))
    hoja_resumen(wb, prov, n)
    _, col0, fin_nac = hoja_nacional(wb, den_nac)
    hoja_enso(wb, enso)
    hoja_poblacion(wb, pob, prov)
    hoja_provincias(wb, prov)
    _, k_ini, k_fin = hoja_kit(wb, kit)
    hoja_demanda(wb, n, k_ini, k_fin)
    hoja_validacion(wb, n, col0, fin_nac, panel)
    hoja_fuentes(wb, manifest)
    hoja_diccionario(wb)
    aplicar_fuente(wb)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.salida)
    print(f"Excel guardado en {args.salida}")


if __name__ == "__main__":
    main()
