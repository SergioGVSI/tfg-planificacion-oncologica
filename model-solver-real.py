# Modelo MILP lexicografico para citacion en hospital de dia oncologico
# Datos reales: instancias .dat del articulo (San Martino Hospital, Genova)
# Carello, Passacantando & Tanfani (EJOR, 2025)
# Secuencia: P1 (min overtime F1) -> P2 (min max-espera F2) -> P3 (max sillon F3)
# Solver: Gurobi via amplpy

import os
import re
import sys
import time
from glob import glob

import numpy as np
import pandas as pd
from amplpy import AMPL, modules

# ─────────────────────────────────────────────────────────────────────────────
# LICENCIA AMPL
# ─────────────────────────────────────────────────────────────────────────────
AMPL_UUID = ""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────────────────────
REAL_DIR      = os.path.join("Databases", "real-sets")
OPTIMIZED_DIR = os.path.join("Databases", "optimized-sets-real")
LOGS_DIR      = "logs-real"

# Articulo: |SV|=48 (36 regulares + 12 overtime), |SI|=66 (54 + 12)
SLOTS_VISITA   = list(range(1, 49))
SLOTS_INFUSION = list(range(1, 67))
NV = 36   # ultimo slot regular visitas  -> 14:00
NI = 54   # ultimo slot regular infusion -> 17:00

MAX_SALAS    = 6
MAX_SILLONES = 26
MAX_CAMAS    = 27

TIME_LIMIT_P1  = 600
TIME_LIMIT_P2  = 600
TIME_LIMIT_P3  = 1200
TIME_LIMIT_DAY = 120
GAP_REL        = 0.05

# Instancias a descartar del barrido principal:
#  - istanza14 (612/156, 4 dias): la unica de las 52 que NO figura en la Tabla 7
#    (cruzando ptot y nº de criticos con las 51 filas). Las 51 restantes se procesan.
EXCLUDED_INSTANCES = {"istanza14"}

# ─────────────────────────────────────────────────────────────────────────────
# MODELO AMPL — con restriccion MCP real (constraint 2 del articulo)
# ─────────────────────────────────────────────────────────────────────────────
# Diferencia clave respecto al solver de datos simulados:
# - Se añaden conjuntos R (salas) y K (patologias)
# - Se añade param alpha{P} (patologia de cada paciente)
# - Se añade param w{R,K,D} (asignacion sala-patologia-dia, MCP)
# - La restriccion cap_salas usa MCP por patologia, no capacidad global
# - D es variable (4 o 5 dias segun la instancia)
AMPL_MODEL = """
# ── Conjuntos ──────────────────────────────────────────────────────────────
set P;          # todos los pacientes
set PB;         # criticos (requieren cama)
set PC;         # no-criticos (sillon o cama)
set D;          # dias de trabajo {1..giornilav}
set SV;         # slots de visita {1..48}
set SI;         # slots de infusion {1..66}
set K;          # patologias {1..7}

# ── Parametros ─────────────────────────────────────────────────────────────
param vp{P}      integer >= 1;   # duracion visita (slots)
param fp{P}      integer >= 1;   # duracion infusion (slots)
param alpha{P}   integer >= 1;   # patologia del paciente
param nsalas{K, D} integer >= 0; # MCP: nº de salas asignadas a patologia k en dia d
param NV         integer;        # ultimo slot regular visita
param NI         integer;        # ultimo slot regular infusion
param RSILL      integer;        # capacidad sillones
param RCAMA      integer;        # capacidad camas

# epsilon-constraint (activados en P2 y P3)
param v1_bar default 1e9;
param v2_bar default 1e9;

# ── Variables de decision ──────────────────────────────────────────────────
var x{P, D, SV}  binary;    # inicio visita
var y{PB, D, SI} binary;    # inicio infusion criticos (cama)
var zB{PC, D, SI} binary;   # inicio infusion no-criticos en cama (fallback)
var zC{PC, D, SI} binary;   # inicio infusion no-criticos en sillon (preferido)
var aV{D} >= 0;              # overtime visitas por dia
var aI{D} >= 0;              # overtime infusiones por dia
var om{D} >= 0;              # max espera por dia

# ── Funciones objetivo ─────────────────────────────────────────────────────
minimize OvertimeTotal:
    sum{d in D} (aV[d] + aI[d]);

minimize WaitTotal:
    sum{d in D} om[d];

maximize ChairsTotal:
    sum{p in PC, d in D, s in SI} zC[p,d,s];

# ── Restricciones ──────────────────────────────────────────────────────────

# (1) Unicidad
subject to unicidad{p in P}:
    sum{d in D, s in SV} x[p,d,s] = 1;

# (2) Capacidad salas por patologia (MCP real — ecuacion 2 del articulo)
# nsalas[k,d] = sum_r w[r,k,d] del articulo (nº de salas asignadas a k en d)
subject to cap_salas{k in K, d in D, s in SV}:
    sum{p in P, q in SV: alpha[p] = k and q >= max(1, s-vp[p]+1) and q <= s} x[p,d,q]
    <= nsalas[k,d];

# (3) Mismo dia visita-infusion criticos
subject to mismo_dia_pb{p in PB, d in D}:
    sum{s in SV} x[p,d,s] = sum{s in SI} y[p,d,s];

# (4) Mismo dia visita-infusion no-criticos
subject to mismo_dia_pc{p in PC, d in D}:
    sum{s in SV} x[p,d,s] = sum{s in SI} (zB[p,d,s] + zC[p,d,s]);

# (5) Secuencialidad criticos
subject to seq_pb{p in PB, d in D}:
    sum{s in SI} s * y[p,d,s] >= sum{s in SV} (s + vp[p]) * x[p,d,s];

# (6) Secuencialidad no-criticos
subject to seq_pc{p in PC, d in D}:
    sum{s in SI} s * (zB[p,d,s] + zC[p,d,s]) >= sum{s in SV} (s + vp[p]) * x[p,d,s];

# (7) Capacidad sillones (ventana deslizante)
subject to cap_sillones{d in D, s in SI}:
    sum{p in PC, q in SI: q >= max(1, s-fp[p]+1) and q <= s} zC[p,d,q] <= RSILL;

# (8) Capacidad camas (ventana deslizante)
subject to cap_camas{d in D, s in SI}:
    sum{p in PB, q in SI: q >= max(1, s-fp[p]+1) and q <= s} y[p,d,q]
  + sum{p in PC, q in SI: q >= max(1, s-fp[p]+1) and q <= s} zB[p,d,q] <= RCAMA;

# (9a) Overtime visitas
subject to ot_visitas{p in P, d in D}:
    aV[d] >= sum{s in SV} (s + vp[p] - 1) * x[p,d,s] - NV;

# (9b) Overtime infusiones criticos
subject to ot_inf_pb{p in PB, d in D}:
    aI[d] >= sum{s in SI} (s + fp[p] - 1) * y[p,d,s] - NI;

# (9c) Overtime infusiones no-criticos
subject to ot_inf_pc{p in PC, d in D}:
    aI[d] >= sum{s in SI} (s + fp[p] - 1) * (zB[p,d,s] + zC[p,d,s]) - NI;

# (10) Espera maxima diaria
subject to espera_pb{p in PB, d in D}:
    om[d] >= sum{s in SI} s * y[p,d,s] - sum{s in SV} (s + vp[p]) * x[p,d,s];

subject to espera_pc{p in PC, d in D}:
    om[d] >= sum{s in SI} s * (zB[p,d,s] + zC[p,d,s]) - sum{s in SV} (s + vp[p]) * x[p,d,s];

# Epsilon-constraints
subject to eps_f1: sum{d in D} (aV[d] + aI[d]) <= v1_bar;
subject to eps_f2: sum{d in D} om[d] <= v2_bar;
"""


# ─────────────────────────────────────────────────────────────────────────────
# TEE
# ─────────────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, file_path):
        self._file   = open(file_path, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._stdout
        self._file.close()

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()


# ─────────────────────────────────────────────────────────────────────────────
# PARSER DE FICHEROS .dat
# ─────────────────────────────────────────────────────────────────────────────
def parse_dat(path):
    """
    Lee un fichero .dat del articulo y devuelve un dict con:
      giornilav  : int (4 o 5)
      w          : dict {(r, k, d): 1}  -- MCP
      patients   : list de dict {id, alpha, v, f, lambda_}
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    # giornilav
    m = re.search(r"param\s+giornilav\s*:=\s*(\d+)", text)
    giornilav = int(m.group(1))

    # c[i,j] = 1 si el i-esimo dia laborable corresponde al j-esimo dia natural.
    # Se construye c_map[i] = j (dia natural de semana del i-esimo dia laborable).
    c_map = {}
    c_block = re.search(r"param\s+c\s*:\s*([\d\s]+?):=(.*?);", text, re.DOTALL)
    if c_block:
        cols = [int(x) for x in c_block.group(1).split()]
        for row in c_block.group(2).strip().splitlines():
            vals = row.split()
            if not vals or not vals[0].isdigit():
                continue
            i = int(vals[0])
            for col_idx, flag in zip(cols, vals[1:]):
                if flag == "1":
                    c_map[i] = col_idx
                    break

    # w[r, k, d]
    w = {}
    w_block = re.search(r"param\s+w\s*:=(.*?);", text, re.DOTALL)
    if w_block:
        for entry in re.finditer(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s+1", w_block.group(1)):
            r, k, d = int(entry.group(1)), int(entry.group(2)), int(entry.group(3))
            w[(r, k, d)] = 1

    # pacientes: bloque "param: alpha v f lambda :="
    patients = []
    pat_block = re.search(r"param\s*:\s*alpha\s+v\s+f\s+lambda\s*:=(.*?);", text, re.DOTALL)
    if pat_block:
        for row in re.finditer(r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", pat_block.group(1)):
            patients.append({
                "id":      int(row.group(1)),
                "alpha":   int(row.group(2)),
                "v":       int(row.group(3)),
                "f":       int(row.group(4)),
                "lambda_": int(row.group(5)),
            })

    return {"giornilav": giornilav, "c_map": c_map, "w": w, "patients": patients}


MAX_SALAS_DIA = 6  # nº de salas fisicas del centro (no se puede exceder por dia)


# ─────────────────────────────────────────────────────────────────────────────
# MCP — OPCION C (sesion2/tfg-fidelidad-datos-y-cota-lb1.md §12.7)
# ─────────────────────────────────────────────────────────────────────────────
# El MCP del articulo se define al inicio de cada MES, dimensionado al maximo semanal
# de solicitudes de ese mes, y se mantiene FIJO (cambia solo cada 3 meses; §5.1, §5.5).
# El MCP de los .dat es ese MCP mensual (la "firma" de bloques por patologia es constante
# en bloques de 4-9 semanas = meses), pero esta MAL BALANCEADO: en los 10 meses infra-asigna
# a la patologia pico (casi siempre HE) y deja deficit -> overtime que el articulo no tiene
# (e infactibilidad en istanza31). Un MCP mensual de 0 deficit SI cabe en las 30 plazas en
# los 10 meses, asi que se RE-BALANCEA una vez por mes (no por semana, que romperia la
# fijeza). En las semanas con festivo se mantiene el MCP mensual fijo y se PIERDE el dia
# festivo (mecanismo del articulo).

def _firma(w):
    """Bloques por patologia (k=1..7) del MCP crudo. Constante dentro de cada mes, por lo
    que sirve para agrupar las semanas en periodos = meses."""
    cnt = {k: 0 for k in range(1, 8)}
    for (_r, k, _d), val in w.items():
        cnt[k] += val
    return tuple(cnt[k] for k in range(1, 8))


def _alloc_salas(demanda_k):
    """Reparte las 30 salas-dia de una semana completa (6 salas x 5 dias) entre las 7
    patologias minimizando el deficit de slots de visita regulares respecto a demanda_k
    (= PICO de demanda del mes). Greedy: cada sala va a la patologia de mayor deficit
    restante; cubierto el deficit, las sobrantes a las de mayor demanda. Optimo porque el
    deficit es separable y la ganancia marginal de cada sala es constante (NV)."""
    salas_k = {k: 0 for k in range(1, 8)}

    def deficit_restante(k):
        return demanda_k.get(k, 0) - NV * salas_k[k]

    for _ in range(MAX_SALAS_DIA * 5):
        con_deficit = [k for k in range(1, 8) if deficit_restante(k) > 0]
        if con_deficit:
            k_best = max(con_deficit, key=deficit_restante)
        else:
            k_best = max(range(1, 8), key=lambda kk: demanda_k.get(kk, 0))
        salas_k[k_best] += 1
    return salas_k


def _optimize_layout(salas_k, weeks):
    """Elige el reparto del MCP mensual FIJO por dia natural (layout[k,d]) que MINIMIZA el
    overtime de visita del periodo, manteniendo fijos los totales por patologia salas_k.

    Es la parte de la Opcion C que el articulo NO publica (su layout concreto, fijado por
    el procedimiento tactico de Carello et al. 2022, que no minimiza overtime). Al ser
    desconocido, elegimos el layout racional: el que minimiza el overtime de visita total
    bajo la restriccion de MCP mensual fijo. Solo afecta a las semanas con festivo (las de
    5 dias usan los 5 dias -> deficit 0 con cualquier layout).

    weeks = lista de (dias_naturales_trabajados:set, demanda_por_patologia:dict) del periodo.
    MILP (Gurobi): min suma de deficits de visita por (patologia, semana), s.a. totales por
    patologia fijos, <= MAX_SALAS_DIA salas/dia, y factibilidad (dem <= |SV|*salas en dias
    trabajados) en cada semana. Devuelve layout[(k, dia_natural)] = nº salas."""
    ampl = AMPL()
    ampl.eval("""
        set K; set DN; set W;
        param salas{K} integer >= 0;
        param dem{K, W} >= 0;
        param worked{W, DN} binary;
        param NVp; param SVmax; param MAXD;
        var L{K, DN} integer >= 0;
        var defi{K, W} >= 0;
        var sobre{K, DN} >= 0;          # bloques de k por encima de 1 en un dia (balanceo)
        # Objetivo: PRIMARIO minimizar overtime de visita del periodo; SECUNDARIO (peso
        # pequeno) esparcir cada patologia entre dias. El balanceo evita layouts concentrados
        # arbitrarios en periodos sin festivo (donde el overtime es 0 con cualquier reparto),
        # que dificultan a P1 encontrar la solucion de 0 overtime.
        minimize Obj:
            1000 * sum{k in K, w in W} defi[k,w] + sum{k in K, d in DN} sobre[k,d];
        subject to total{k in K}: sum{d in DN} L[k,d] = salas[k];
        subject to cap{d in DN}:  sum{k in K} L[k,d] <= MAXD;
        subject to defc{k in K, w in W}:
            defi[k,w] >= dem[k,w] - NVp * sum{d in DN} worked[w,d]*L[k,d];
        subject to feas{k in K, w in W}:
            SVmax * sum{d in DN} worked[w,d]*L[k,d] >= dem[k,w];
        subject to spread{k in K, d in DN}: sobre[k,d] >= L[k,d] - 1;
    """)
    K = list(range(1, 8)); DN = list(range(1, 6)); W = list(range(len(weeks)))
    ampl.set["K"] = K; ampl.set["DN"] = DN; ampl.set["W"] = W
    ampl.param["salas"]  = {k: salas_k[k] for k in K}
    ampl.param["NVp"]    = NV
    ampl.param["SVmax"]  = len(SLOTS_VISITA)   # |SV| = 48 (techo absoluto/sala/dia)
    ampl.param["MAXD"]   = MAX_SALAS_DIA
    ampl.param["dem"]    = {(k, w): weeks[w][1].get(k, 0) for k in K for w in W}
    ampl.param["worked"] = {(w, d): (1 if d in weeks[w][0] else 0)
                            for w in W for d in DN}
    ampl.option["solver"] = "gurobi"
    ampl.option["gurobi_options"] = "outlev=0 mipgap=0"
    ampl.solve()

    if "infeasible" in ampl.get_value("solve_result").lower():
        raise RuntimeError("layout mensual infactible: la demanda de algun periodo no cabe "
                           "en 30 salas-dia tras perder el dia festivo")

    df = ampl.var["L"].get_values().to_pandas().reset_index()
    layout = {(k, d): 0 for k in K for d in DN}
    for row in df.values:
        layout[(int(row[0]), int(row[1]))] = int(round(row[-1]))
    return layout


def build_period_mcps(dat_paths):
    """OPCION C: reconstruye un MCP MENSUAL FIJO por periodo (firma del MCP crudo constante
    = un mes), dimensionado al PICO de demanda del periodo, con reparto por dia natural fijo.
    Es el metodo del propio articulo (MCP mensual al maximo semanal, fijo) pero re-balanceado.

    Devuelve {stem: layout} con layout[(k, dia_natural)] = nº salas, el MISMO para todas las
    semanas del mes. En cada semana se descartan luego los bloques del dia festivo."""
    info = []
    for path in dat_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        dat = parse_dat(path)
        dem = {k: 0 for k in range(1, 8)}
        for p in dat["patients"]:
            dem[p["alpha"]] = dem.get(p["alpha"], 0) + p["v"]
        g = dat["giornilav"]
        c_map = dat["c_map"] or {i: i for i in range(1, g + 1)}
        festivos = set(range(1, 6)) - set(c_map.values())
        info.append({"stem": stem, "firma": _firma(dat["w"]),
                     "dem": dem, "festivos": festivos})

    # agrupar por firma consecutiva (= mes)
    periodos = []
    for it in info:
        if not periodos or periodos[-1]["firma"] != it["firma"]:
            periodos.append({"firma": it["firma"], "items": []})
        periodos[-1]["items"].append(it)

    layout_by_stem = {}
    for per in periodos:
        maxdem  = {k: max(it["dem"][k] for it in per["items"]) for k in range(1, 8)}
        salas_k = _alloc_salas(maxdem)
        # semanas del periodo: (dias naturales trabajados, demanda por patologia)
        weeks  = [(set(range(1, 6)) - it["festivos"], it["dem"]) for it in per["items"]]
        layout = _optimize_layout(salas_k, weeks)
        for it in per["items"]:
            layout_by_stem[it["stem"]] = layout
    return layout_by_stem


def _nsalas_from_layout(layout, g, c_map):
    """Convierte el layout mensual (por dia natural) en nsalas[(k, dia_laborable)] de una
    semana, descartando los bloques del dia FESTIVO (los dias naturales no trabajados)."""
    nsalas = {(k, lab): 0 for k in range(1, 8) for lab in range(1, g + 1)}
    for lab in range(1, g + 1):
        d_nat = c_map[lab]
        for k in range(1, 8):
            nsalas[(k, lab)] = layout.get((k, d_nat), 0)
    return nsalas


def prepare_instance(dat, monthly_layout):
    """Convierte parse_dat en las estructuras del solver. nsalas se construye a partir del
    MCP MENSUAL FIJO del periodo (Opcion C, build_period_mcps), descartando el dia festivo
    en las semanas de 4 dias."""
    g    = dat["giornilav"]
    dias = list(range(1, g + 1))
    # c_map por defecto = identidad (semana sin festivos)
    c_map = dat["c_map"] or {i: i for i in dias}

    nsalas = _nsalas_from_layout(monthly_layout, g, c_map)

    PB, PC = [], []
    v_p, f_p, alpha_p = {}, {}, {}

    for p in dat["patients"]:
        pid = p["id"]
        v_p[pid]    = p["v"]
        f_p[pid]    = p["f"]
        alpha_p[pid] = p["alpha"]
        if p["lambda_"] == 1:
            PB.append(pid)
        else:
            PC.append(pid)

    return dias, PB, PC, v_p, f_p, alpha_p, nsalas


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCION DE LA INSTANCIA AMPL
# ─────────────────────────────────────────────────────────────────────────────
def _build_ampl(dias, PB, PC, v_p, f_p, alpha_p, nsalas):
    ampl = AMPL()
    ampl.eval(AMPL_MODEL)

    P = PB + PC
    patologias = list(range(1, 8))

    ampl.set["P"]  = P
    ampl.set["PB"] = PB
    ampl.set["PC"] = PC
    ampl.set["D"]  = dias
    ampl.set["SV"] = SLOTS_VISITA
    ampl.set["SI"] = SLOTS_INFUSION
    ampl.set["K"]  = patologias

    ampl.param["NV"]   = NV
    ampl.param["NI"]   = NI
    ampl.param["RSILL"] = MAX_SILLONES
    ampl.param["RCAMA"] = MAX_CAMAS

    ampl.param["vp"]    = {p: v_p[p]    for p in P}
    ampl.param["fp"]    = {p: f_p[p]    for p in P}
    ampl.param["alpha"] = {p: alpha_p[p] for p in P}

    # nsalas[k, d] = nº de salas para la patologia k en el dia laborable d
    ampl.param["nsalas"] = {(k, d): nsalas.get((k, d), 0)
                            for k in patologias for d in dias}

    ampl.option["solver"]         = "gurobi"
    ampl.option["gurobi_options"] = f"mipgap={GAP_REL} outlev=1"

    return ampl


def _set_time_limit(ampl, seconds):
    ampl.option["gurobi_options"] = f"mipgap={GAP_REL} outlev=1 timelim={seconds}"


def _solve_status(ampl):
    return ampl.get_value("solve_result")


def _obj_value(ampl, obj_name):
    try:
        return ampl.get_objective(obj_name).value()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LECTURA DE SOLUCION
# ─────────────────────────────────────────────────────────────────────────────
def _read_var3_df(df):
    result = {}
    arr = df.values
    for row in arr:
        result[(row[0], int(row[1]), int(row[2]))] = row[3]
    return result


def _read_x(ampl):
    df = ampl.var["x"].get_values().to_pandas().reset_index()
    return _read_var3_df(df)


def _read_var3(ampl, var_name, index_set):
    if not index_set:
        return {}
    df = ampl.var[var_name].get_values().to_pandas().reset_index()
    return _read_var3_df(df)


def _day_assignments(x_vals, P, dias):
    Pd = {d: [] for d in dias}
    assigned = set()
    for (p, d, s), v in x_vals.items():
        if v > 0.5 and p not in assigned:
            Pd[d].append(p)
            assigned.add(p)
    return Pd


# ─────────────────────────────────────────────────────────────────────────────
# LOWER BOUND LB1 — Procedimiento 1 del articulo (§4.1)
# ─────────────────────────────────────────────────────────────────────────────
def compute_lb1(dias, PB, PC, v_p, f_p, alpha_p, nsalas):
    """Calcula una cota inferior VALIDA del overtime F1 (desigualdad valida F1 >= LB1).

    Procedimiento 1 del articulo (§4.1), con la correccion verificada en
    sesion2/tfg-fidelidad-datos-y-cota-lb1.md §9:

      - El articulo define LB1 = optimo de la relajacion continua (r1) de P1, restringida
        a los pacientes de las patologias en DEFICIT (NV*salas_k < dem_k), con sillones y
        camas ilimitados. Esa cota es VALIDA (nunca recorta el optimo) pero, tal cual,
        sale 0 en las 51 instancias: la restriccion de overtime (9a) es POR PACIENTE y el
        LP esparce las visitas fraccionadas para que ningun paciente la viole (cota vacua).

      - ARREGLO: se refuerza (r1) con una desigualdad VALIDA agregada por (patologia, dia)
            aV[d] >= ( sum_{p: alpha=k} vp[p] * sum_s x[p,d,s] ) / nsalas[k,d] - NV
        que captura el argumento de capacidad sin la dilucion fraccionaria: la sala mas
        cargada de la patologia k el dia d acaba en un slot >= carga/nsalas[k,d]. LB1 es el
        optimo de (r1) reforzado, y es la cota que se impone como F1 >= LB1.

    lb_conteo (= suma de deficits por patologia) NO es una cota valida (F1 es suma de
    MAXIMOS diarios, no de citas en overtime; las salas de una patologia desbordan en
    PARALELO y el maximo entre patologias del mismo dia absorbe los menores), por lo que
    puede SUPERAR el optimo real y recortarlo. Se calcula SOLO como cross-check de log;
    NUNCA se impone como corte.

    Devuelve (lb1, info) con detalles para el log.
    """
    P = PB + PC

    # Patologias en deficit y demanda/capacidad por patologia
    salas_k = {k: sum(nsalas.get((k, d), 0) for d in dias) for k in range(1, 8)}
    dem_k   = {k: 0 for k in range(1, 8)}
    for p in P:
        dem_k[alpha_p[p]] += v_p[p]
    K_bar = [k for k in range(1, 8) if NV * salas_k[k] < dem_k[k]]

    if not K_bar:
        return 0.0, {"K_bar": [], "n_pbar": 0, "lb_conteo": 0.0, "lb_lp": 0.0}

    # Cross-check informativo (NO valido, NO se impone): suma de deficits por patologia
    lb_conteo = sum(max(0, dem_k[k] - NV * salas_k[k]) for k in K_bar)

    # Cota LP REFORZADA: relajacion continua (r1) sobre los pacientes en deficit, con la
    # desigualdad valida agregada por (patologia, dia) que evita la dilucion fraccionaria.
    P_bar  = [p for p in P if alpha_p[p] in K_bar]
    PB_bar = [p for p in P_bar if p in PB]
    PC_bar = [p for p in P_bar if p in PC]

    ampl = _build_ampl(dias, PB_bar, PC_bar, v_p, f_p, alpha_p, nsalas)
    ampl.eval("drop eps_f1; drop eps_f2;")
    ampl.param["RSILL"] = 10 ** 6   # camas/sillones ilimitados (la cota mira las VISITAS)
    ampl.param["RCAMA"] = 10 ** 6
    # Desigualdad valida agregada que refuerza la (9a) por-paciente (que el LP diluiria):
    ampl.eval(
        "subject to lb1_vis_agg{k in K, d in D: nsalas[k,d] >= 1}:"
        " aV[d] >= (sum{p in P, s in SV: alpha[p] = k} vp[p]*x[p,d,s]) / nsalas[k,d] - NV;"
    )
    ampl.option["relax_integrality"] = 1
    ampl.eval("objective OvertimeTotal;")
    ampl.option["gurobi_options"] = "outlev=0"
    ampl.solve()
    lb_lp = _obj_value(ampl, "OvertimeTotal")
    lb_lp = max(0.0, lb_lp) if lb_lp is not None else 0.0

    # LB1 = cota LP reforzada (VALIDA). lb_conteo solo como cross-check (ver docstring).
    lb1 = lb_lp
    info = {"K_bar": K_bar, "n_pbar": len(P_bar),
            "lb_conteo": lb_conteo, "lb_lp": lb_lp}
    return lb1, info


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEMA 1 — min F1
# ─────────────────────────────────────────────────────────────────────────────
def solve_p1(dias, PB, PC, v_p, f_p, alpha_p, w_mcp):
    P = PB + PC
    print(f"\n{'─' * 60}")
    print(f"[P1] min overtime F1 | {len(PB)} criticos  {len(PC)} no-criticos")

    # Lower bound LB1 (Procedimiento 1): desigualdad valida F1 >= LB1 para acelerar P1
    lb1, lb1_info = compute_lb1(dias, PB, PC, v_p, f_p, alpha_p, w_mcp)
    _xchk = " [!] conteo>LB1: cota de conteo NO valida, se ignora" \
        if lb1_info['lb_conteo'] > lb1 + 1e-6 else ""
    print(f"[P1] LB1 = {lb1:.1f} (LP reforzado)  "
          f"[conteo_xcheck={lb1_info['lb_conteo']:.1f}] | deficit={lb1_info['K_bar']} "
          f"{lb1_info['n_pbar']} pac{_xchk}")

    ampl = _build_ampl(dias, PB, PC, v_p, f_p, alpha_p, w_mcp)
    ampl.eval("drop eps_f1; drop eps_f2;")
    ampl.eval("objective OvertimeTotal;")
    # Desigualdad valida F1 >= LB1 (cota inferior del overtime)
    if lb1 > 0:
        ampl.eval(f"subject to lb1_cut: sum{{d in D}} (aV[d] + aI[d]) >= {lb1};")
    _set_time_limit(ampl, TIME_LIMIT_P1)

    t0 = time.time()
    ampl.solve()
    elapsed = time.time() - t0

    status = _solve_status(ampl)
    f1     = _obj_value(ampl, "OvertimeTotal")
    print(f"[P1] {status} | F1 = {f1:.1f} slots overtime | {elapsed:.1f}s")

    x_vals = _read_x(ampl)
    Pd     = _day_assignments(x_vals, P, dias)

    aV_vals = {d: max(0.0, ampl.var["aV"][d].value()) for d in dias}
    aI_vals = {d: max(0.0, ampl.var["aI"][d].value()) for d in dias}

    return {
        "status":  status,
        "v1":      f1 if f1 is not None else float("inf"),
        "elapsed": elapsed,
        "Pd":      Pd,
        "aV":      aV_vals,
        "aI":      aI_vals,
        "ampl":    ampl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DESCOMPOSICION DIARIA — Procedimiento 2 (warm start para P2)
# ─────────────────────────────────────────────────────────────────────────────
def _solve_day_p2(d, PB_d, PC_d, v_p, f_p, alpha_p, w_mcp, aV_fixed, aI_fixed):
    P_d = PB_d + PC_d
    if not P_d:
        return {"om": 0.0, "x": {}, "y": {}, "zB": {}, "zC": {}}

    ampl = _build_ampl([d], PB_d, PC_d, v_p, f_p, alpha_p, w_mcp)
    ampl.var["aV"][d].fix(aV_fixed)
    ampl.var["aI"][d].fix(aI_fixed)

    ampl.eval("drop eps_f1; drop eps_f2;")
    ampl.eval("objective WaitTotal;")
    ampl.option["gurobi_options"] = f"mipgap={GAP_REL} outlev=0 timelim={TIME_LIMIT_DAY}"

    ampl.solve()

    status = _solve_status(ampl)
    if "infeasible" in status.lower():
        print(f"  [WARN] sub-problema dia {d} infactible — warm start parcial")
        return {"om": 0.0, "x": {}, "y": {}, "zB": {}, "zC": {}}

    om_val = max(0.0, ampl.var["om"][d].value())
    return {
        "om":  om_val,
        "x":   _read_x(ampl),
        "y":   _read_var3(ampl, "y",  PB_d),
        "zB":  _read_var3(ampl, "zB", PC_d),
        "zC":  _read_var3(ampl, "zC", PC_d),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEMA 2 — min F2
# ─────────────────────────────────────────────────────────────────────────────
def solve_p2(dias, PB, PC, v_p, f_p, alpha_p, w_mcp, r1):
    P   = PB + PC
    v1  = r1["v1"]
    Pd  = r1["Pd"]
    aV0 = r1["aV"]
    aI0 = r1["aI"]

    print(f"\n{'─' * 60}")
    print(f"[P2] min espera maxima diaria F2 | v1_bar = {v1:.1f} slots")
    print("[P2] Procedimiento 2: descomposicion por dia...")

    day_sols = {}
    for d in dias:
        PB_d = [p for p in Pd[d] if p in PB]
        PC_d = [p for p in Pd[d] if p in PC]
        s    = _solve_day_p2(d, PB_d, PC_d, v_p, f_p, alpha_p, w_mcp, aV0[d], aI0[d])
        day_sols[d] = s
        print(f"  Dia {d}: {len(Pd[d])} pacientes | omega = {s['om']:.1f} slots")

    ampl = _build_ampl(dias, PB, PC, v_p, f_p, alpha_p, w_mcp)
    ampl.eval("drop eps_f2;")
    ampl.param["v1_bar"] = v1
    ampl.eval("objective WaitTotal;")
    _set_time_limit(ampl, TIME_LIMIT_P2)

    for d, sol in day_sols.items():
        for (p, dd, s), v in sol["x"].items():
            ampl.var["x"][p, dd, s].set_value(round(v))
        for (p, dd, s), v in sol["y"].items():
            ampl.var["y"][p, dd, s].set_value(round(v))
        for (p, dd, s), v in sol["zB"].items():
            ampl.var["zB"][p, dd, s].set_value(round(v))
        for (p, dd, s), v in sol["zC"].items():
            ampl.var["zC"][p, dd, s].set_value(round(v))
        ampl.var["om"][d].set_value(sol["om"])
    for d in dias:
        ampl.var["aV"][d].set_value(aV0[d])
        ampl.var["aI"][d].set_value(aI0[d])

    print("[P2] Resolviendo problema completo con warm start...")
    t0 = time.time()
    ampl.solve()
    elapsed = time.time() - t0

    status = _solve_status(ampl)
    f2     = _obj_value(ampl, "WaitTotal")
    print(f"[P2] {status} | F2 = {f2:.1f} slots | {elapsed:.1f}s")

    x_vals = _read_x(ampl)
    Pd2    = _day_assignments(x_vals, P, dias)

    return {
        "status":  status,
        "v2":      f2 if f2 is not None else float("inf"),
        "elapsed": elapsed,
        "Pd":      Pd2,
        "aV":      {d: max(0.0, ampl.var["aV"][d].value()) for d in dias},
        "aI":      {d: max(0.0, ampl.var["aI"][d].value()) for d in dias},
        "om":      {d: max(0.0, ampl.var["om"][d].value())  for d in dias},
        "ampl":    ampl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEMA 3 — max F3
# ─────────────────────────────────────────────────────────────────────────────
def solve_p3(dias, PB, PC, v_p, f_p, alpha_p, w_mcp, r1, r2):
    P  = PB + PC
    v1 = r1["v1"]
    v2 = r2["v2"]
    a2 = r2["ampl"]

    print(f"\n{'─' * 60}")
    print(f"[P3] max sillones F3 | v1_bar = {v1:.1f}  v2_bar = {v2:.1f}")

    ampl = _build_ampl(dias, PB, PC, v_p, f_p, alpha_p, w_mcp)
    ampl.param["v1_bar"] = v1
    ampl.param["v2_bar"] = v2
    ampl.eval("objective ChairsTotal;")
    _set_time_limit(ampl, TIME_LIMIT_P3)

    for p in P:
        for d in dias:
            for s in SLOTS_VISITA:
                try:
                    ampl.var["x"][p, d, s].set_value(round(a2.var["x"][p, d, s].value() or 0))
                except Exception:
                    pass
    for p in PB:
        for d in dias:
            for s in SLOTS_INFUSION:
                try:
                    ampl.var["y"][p, d, s].set_value(round(a2.var["y"][p, d, s].value() or 0))
                except Exception:
                    pass
    for p in PC:
        for d in dias:
            for s in SLOTS_INFUSION:
                try:
                    ampl.var["zB"][p, d, s].set_value(round(a2.var["zB"][p, d, s].value() or 0))
                    ampl.var["zC"][p, d, s].set_value(round(a2.var["zC"][p, d, s].value() or 0))
                except Exception:
                    pass
    for d in dias:
        ampl.var["aV"][d].set_value(r2["aV"][d])
        ampl.var["aI"][d].set_value(r2["aI"][d])
        ampl.var["om"][d].set_value(r2["om"][d])

    t0 = time.time()
    ampl.solve()
    elapsed = time.time() - t0

    status = _solve_status(ampl)
    f3     = _obj_value(ampl, "ChairsTotal")
    print(f"[P3] {status} | F3 = {f3:.0f} pacientes en sillon | {elapsed:.1f}s")

    return {
        "status":  status,
        "f3":      f3 if f3 is not None else 0.0,
        "elapsed": elapsed,
        "aV":      {d: max(0.0, ampl.var["aV"][d].value()) for d in dias},
        "aI":      {d: max(0.0, ampl.var["aI"][d].value()) for d in dias},
        "om":      {d: max(0.0, ampl.var["om"][d].value())  for d in dias},
        "ampl":    ampl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCION DE AGENDA FINAL
# ─────────────────────────────────────────────────────────────────────────────
def _extract_agenda(ampl3, dias, PB, PC, v_p):
    rows     = []
    complete = True
    PB_set   = set(PB)

    x_vals  = _read_x(ampl3)
    y_vals  = _read_var3(ampl3, "y",  PB)
    zC_vals = _read_var3(ampl3, "zC", PC)
    zB_vals = _read_var3(ampl3, "zB", PC)

    for p in PB + PC:
        fila    = {"ID_Paciente": p, "Critico": 1 if p in PB_set else 0}
        found_v = False
        found_i = False

        for d in dias:
            for s in SLOTS_VISITA:
                if (x_vals.get((p, d, s), 0) or 0) > 0.5:
                    fila["Dia"]                = d
                    fila["Slot_Inicio_Visita"] = s
                    found_v = True

                    if p in PB_set:
                        for si in SLOTS_INFUSION:
                            if (y_vals.get((p, d, si), 0) or 0) > 0.5:
                                fila["Slot_Inicio_Infusion"] = si
                                fila["Recurso_Infusion"]     = "cama"
                                found_i = True
                                break
                    else:
                        for si in SLOTS_INFUSION:
                            if (zC_vals.get((p, d, si), 0) or 0) > 0.5:
                                fila["Slot_Inicio_Infusion"] = si
                                fila["Recurso_Infusion"]     = "sillon"
                                found_i = True
                                break
                            if (zB_vals.get((p, d, si), 0) or 0) > 0.5:
                                fila["Slot_Inicio_Infusion"] = si
                                fila["Recurso_Infusion"]     = "cama"
                                found_i = True
                                break
                    break
            if found_v:
                break

        if not found_v or not found_i:
            complete = False
            continue

        espera_s               = fila["Slot_Inicio_Infusion"] - (fila["Slot_Inicio_Visita"] + v_p[p])
        fila["Espera_Slots"]   = espera_s
        fila["Espera_Minutos"] = espera_s * 10
        rows.append(fila)

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df, complete


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLETO: P1 -> P2 -> P3
# ─────────────────────────────────────────────────────────────────────────────
def build_and_solve(dias, PB, PC, v_p, f_p, alpha_p, w_mcp):
    n = len(PB) + len(PC)
    print(f"\nPacientes totales: {n}  ({len(PB)} criticos / {len(PC)} no-criticos)")
    print(f"Dias de trabajo: {dias}")

    t_start = time.time()

    r1 = solve_p1(dias, PB, PC, v_p, f_p, alpha_p, w_mcp)
    if r1["v1"] == float("inf"):
        print("[ERROR] P1 no encontro solucion factible.")
        return {"feasible": False}

    r2 = solve_p2(dias, PB, PC, v_p, f_p, alpha_p, w_mcp, r1)
    if r2["v2"] == float("inf"):
        print("[ERROR] P2 no encontro solucion factible.")
        return {"feasible": False}

    r3 = solve_p3(dias, PB, PC, v_p, f_p, alpha_p, w_mcp, r1, r2)

    t_total = time.time() - t_start

    df_agenda, complete = _extract_agenda(r3["ampl"], dias, PB, PC, v_p)

    coherent   = False
    mean_wait  = None
    pct_sillon = 0.0

    if complete and len(df_agenda) == n and not df_agenda.empty:
        if (df_agenda["Espera_Slots"] >= 0).all():
            coherent  = True
            mean_wait = float(df_agenda["Espera_Minutos"].mean())

    if not df_agenda.empty and "Recurso_Infusion" in df_agenda.columns and PC:
        n_sillon   = (df_agenda["Recurso_Infusion"] == "sillon").sum()
        pct_sillon = 100.0 * n_sillon / len(PC)

    f1_real = sum(r3["aV"].values()) + sum(r3["aI"].values())
    f2_real = sum(r3["om"].values())
    f3_real = r3["f3"]

    print(f"\n{'=' * 60}")
    print("RESUMEN SOLUCION FINAL")
    print(f"  F1 (overtime total)      : {f1_real:.1f} slots  ({f1_real * 10:.0f} min)")
    print(f"  F2 (suma max-espera/dia) : {f2_real:.1f} slots")
    print(f"  F3 (no-crit en sillon)   : {f3_real:.0f}  ({pct_sillon:.1f}% de no-criticos)")
    if mean_wait is not None:
        print(f"  Espera media pacientes   : {mean_wait:.2f} min")
    else:
        print("  Espera media pacientes   : N/A")
    print(f"  Coherente                : {coherent}")
    print(f"  Tiempo total             : {t_total:.1f}s  "
          f"(P1={r1['elapsed']:.1f}  P2={r2['elapsed']:.1f}  P3={r3['elapsed']:.1f})")

    return {
        "feasible":   coherent,
        "coherent":   coherent,
        "f1":         f1_real,
        "f2":         f2_real,
        "f3":         f3_real,
        "mean_wait":  mean_wait,
        "pct_sillon": pct_sillon,
        "n_patients": n,
        "n_critical": len(PB),
        "status_p1":  r1["status"],
        "status_p2":  r2["status"],
        "status_p3":  r3["status"],
        "elapsed_p1": r1["elapsed"],
        "elapsed_p2": r2["elapsed"],
        "elapsed_p3": r3["elapsed"],
        "elapsed":    t_total,
        "results_df": df_agenda,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GUARDAR AGENDA
# ─────────────────────────────────────────────────────────────────────────────
def save_agenda(df, stem):
    out_path = os.path.join(OPTIMIZED_DIR, f"agenda_{stem}.csv")
    df.sort_values(["Dia", "Slot_Inicio_Visita"]).to_csv(out_path, index=False)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# BUCLE PRINCIPAL — todas las instancias reales
# ─────────────────────────────────────────────────────────────────────────────
def _natural_key(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 10 ** 9


def run_all():
    modules.activate(AMPL_UUID)

    os.makedirs(OPTIMIZED_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    dat_files = sorted(glob(os.path.join(REAL_DIR, "*.dat")), key=_natural_key)
    if not dat_files:
        print(f"No se encontraron ficheros .dat en {REAL_DIR}.")
        return

    print(f"Instancias encontradas: {len(dat_files)}")

    # Opcion C: MCP mensual fijo, reconstruido en una pre-pasada por TODAS las instancias
    # (la agrupacion en meses necesita ver todas las semanas; el pico es por periodo).
    period_mcp = build_period_mcps(dat_files)

    resumen = []

    for dat_path in dat_files:
        stem = os.path.splitext(os.path.basename(dat_path))[0]

        # El articulo trabaja con 51 instancias. Cruzando (ptot, nº criticos) con la
        # Tabla 7, la unica de nuestras 52 que el articulo descarta es istanza14.
        # El resto -incluida istanza52, de 3 dias- SI estan en la Tabla 7.
        if stem in EXCLUDED_INSTANCES:
            print(f"[SKIP] {stem}: descartada (no figura en la Tabla 7 del articulo).")
            continue

        log_path = os.path.join(LOGS_DIR, f"run_{stem}.txt")

        with Tee(log_path):
            print("=" * 70)
            print(f"INSTANCIA : {stem}  [datos reales, AMPL + Gurobi]")
            print(f"FICHERO   : {dat_path}")
            print(f"FECHA     : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)

            dat = parse_dat(dat_path)
            dias, PB, PC, v_p, f_p, alpha_p, w_mcp = prepare_instance(dat, period_mcp[stem])

            print(f"giornilav = {dat['giornilav']}  |  ptot = {len(dat['patients'])}")
            print(f"Entradas MCP (w): {len(w_mcp)}")

            res = build_and_solve(dias, PB, PC, v_p, f_p, alpha_p, w_mcp)

            out_path = None
            if res.get("feasible"):
                out_path = save_agenda(res["results_df"], stem)
                print(f"Agenda guardada: {out_path}")
            else:
                print("[WARN] Instancia sin solucion coherente — no se guarda agenda.")

        resumen.append({
            "instancia":          stem,
            "giornilav":          dat["giornilav"],
            "n_pacientes":        res.get("n_patients",  0),
            "n_criticos":         res.get("n_critical",  0),
            "status_p1":          res.get("status_p1",   "N/A"),
            "status_p2":          res.get("status_p2",   "N/A"),
            "status_p3":          res.get("status_p3",   "N/A"),
            "f1_overtime_slots":  res.get("f1",          float("nan")),
            "f2_maxwait_slots":   res.get("f2",          float("nan")),
            "f3_sillones":        res.get("f3",          float("nan")),
            "pct_sillon":         res.get("pct_sillon",  float("nan")),
            "espera_media_min":   res.get("mean_wait",   float("nan")),
            "t_p1_s":             res.get("elapsed_p1",  float("nan")),
            "t_p2_s":             res.get("elapsed_p2",  float("nan")),
            "t_p3_s":             res.get("elapsed_p3",  float("nan")),
            "t_total_s":          res.get("elapsed",     float("nan")),
            "coherente":          res.get("coherent",    False),
            "log":                log_path,
            "agenda":             out_path or "",
        })

    df_res   = pd.DataFrame(resumen)
    res_path = os.path.join(OPTIMIZED_DIR, "resumen_ejecucion_real.csv")
    df_res.to_csv(res_path, index=False)
    print(f"\nResumen global guardado en: {res_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_all()
