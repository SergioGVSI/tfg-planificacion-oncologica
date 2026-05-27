# Modelo MILP lexicografico para citacion en hospital de dia oncologico
# Implementacion fiel a Carello, Passacantando & Tanfani (EJOR, 2025)
# Secuencia: P1 (min overtime F1) -> P2 (min max-espera F2) -> P3 (max sillon F3)
# Solver: Gurobi via amplpy

import math
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
AMPL_UUID = "LICENSE_UUID_AQUI"  # Reemplaza con tu UUID de estudiante AMPL

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────────────────────
SIMULATED_DIR = os.path.join("Databases", "simulated-sets")
OPTIMIZED_DIR = os.path.join("Databases", "optimized-sets")
LOGS_DIR      = "logs-ampl"
BASE_SEED     = 20260427

DIAS = [1, 2, 3, 4, 5]

# Articulo: |SV|=48 (36 regulares 8:00-14:00 + 12 overtime), |SI|=66 (54 + 12)
SLOTS_VISITA   = list(range(1, 49))
SLOTS_INFUSION = list(range(1, 67))
NV = 36   # ultimo slot regular visitas  -> 14:00
NI = 54   # ultimo slot regular infusion -> 17:00

MAX_SALAS    = 6
MAX_SILLONES = 26
MAX_CAMAS    = 27

TIME_LIMIT_P1  = 600    # seg
TIME_LIMIT_P2  = 600
TIME_LIMIT_P3  = 1200
TIME_LIMIT_DAY = 120
GAP_REL        = 0.05

FRACCIONES = [
    ("1of3", 1 / 3),
    ("2of3", 2 / 3),
    ("3of3", 1.0),
]

# ─────────────────────────────────────────────────────────────────────────────
# MODELO AMPL — definicion algebraica compartida por P1, P2, P3
# ─────────────────────────────────────────────────────────────────────────────
# El modelo se escribe una vez en sintaxis AMPL. Los tres problemas comparten
# variables y restricciones; lo unico que cambia es la funcion objetivo activa
# y las restricciones epsilon-constraint que se anaden en P2 y P3.
AMPL_MODEL = """
# ── Conjuntos ──────────────────────────────────────────────────────────────
set P;          # todos los pacientes
set PB;         # criticos (requieren cama)
set PC;         # no-criticos (sillon o cama)
set D;          # dias {1..5}
set SV;         # slots de visita {1..48}
set SI;         # slots de infusion {1..66}

# ── Parametros ─────────────────────────────────────────────────────────────
param vp{P}  integer >= 1;   # duracion visita (slots)
param fp{P}  integer >= 1;   # duracion infusion (slots)
param NV     integer;        # ultimo slot regular visita
param NI     integer;        # ultimo slot regular infusion
param RSALAS integer;        # capacidad salas
param RSILL  integer;        # capacidad sillones
param RCAMA  integer;        # capacidad camas

# epsilon-constraint (activados en P2 y P3)
param v1_bar default 1e9;    # cota F1 (fijada por P1)
param v2_bar default 1e9;    # cota F2 (fijada por P2)

# ── Variables de decision ──────────────────────────────────────────────────
var x{P, D, SV}  binary;    # inicio visita
var y{PB, D, SI} binary;    # inicio infusion criticos (cama)
var zB{PC, D, SI} binary;   # inicio infusion no-criticos en cama (fallback)
var zC{PC, D, SI} binary;   # inicio infusion no-criticos en sillon (preferido)
var aV{D} >= 0;              # overtime visitas por dia
var aI{D} >= 0;              # overtime infusiones por dia
var om{D} >= 0;              # max espera por dia

# ── Funciones objetivo (se activa una segun el problema) ───────────────────
minimize OvertimeTotal:
    sum{d in D} (aV[d] + aI[d]);

minimize WaitTotal:
    sum{d in D} om[d];

maximize ChairsTotal:
    sum{p in PC, d in D, s in SI} zC[p,d,s];

# ── Restricciones (1)-(10) del articulo ────────────────────────────────────

# (1) Unicidad
subject to unicidad{p in P}:
    sum{d in D, s in SV} x[p,d,s] = 1;

# (2) Capacidad salas (ventana deslizante)
subject to cap_salas{d in D, s in SV}:
    sum{p in P, q in SV: q >= max(1, s-vp[p]+1) and q <= s} x[p,d,q] <= RSALAS;

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

# Epsilon-constraints (inactivas por defecto, se activan en P2 y P3)
subject to eps_f1: sum{d in D} (aV[d] + aI[d]) <= v1_bar;
subject to eps_f2: sum{d in D} om[d] <= v2_bar;
"""


# ─────────────────────────────────────────────────────────────────────────────
# TEE — stdout a consola y fichero simultaneamente
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
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────
def _natural_key(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 10 ** 9


def get_csv_files():
    files = glob(os.path.join(SIMULATED_DIR, "*.csv"))
    files.sort(key=_natural_key)
    return files


def sample_prefixes(df, seed):
    n    = len(df)
    rng  = np.random.default_rng(seed)
    df_p = df.iloc[rng.permutation(n)].reset_index(drop=True)
    n1, n2 = math.ceil(n / 3), math.ceil(2 * n / 3)
    return {
        "1of3": df_p.iloc[:n1].copy(),
        "2of3": df_p.iloc[:n2].copy(),
        "3of3": df_p.copy(),
    }


def _prepare_data(df):
    ids  = df["ID_Paciente"].tolist()
    v_p  = dict(zip(df["ID_Paciente"], df["Visita_Slots"].astype(int)))
    f_p  = dict(zip(df["ID_Paciente"], df["Infusion_Slots"].astype(int)))
    crit = dict(zip(df["ID_Paciente"], df["Critico_Cama"].astype(int)))
    PB   = [p for p in ids if crit[p] == 1]
    PC   = [p for p in ids if crit[p] == 0]
    return PB, PC, v_p, f_p


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCION DE LA INSTANCIA AMPL
# ─────────────────────────────────────────────────────────────────────────────
def _build_ampl(PB, PC, v_p, f_p):
    """Crea un objeto AMPL, carga el modelo y rellena los datos de la instancia."""
    ampl = AMPL()
    ampl.eval(AMPL_MODEL)

    P = PB + PC

    # Conjuntos
    ampl.set["P"]  = P
    ampl.set["PB"] = PB
    ampl.set["PC"] = PC
    ampl.set["D"]  = DIAS
    ampl.set["SV"] = SLOTS_VISITA
    ampl.set["SI"] = SLOTS_INFUSION

    # Parametros escalares
    ampl.param["NV"]     = NV
    ampl.param["NI"]     = NI
    ampl.param["RSALAS"] = MAX_SALAS
    ampl.param["RSILL"]  = MAX_SILLONES
    ampl.param["RCAMA"]  = MAX_CAMAS

    # Parametros indexados por paciente
    ampl.param["vp"] = {p: v_p[p] for p in P}
    ampl.param["fp"] = {p: f_p[p] for p in P}

    # Configuracion de Gurobi
    ampl.option["solver"]        = "gurobi"
    ampl.option["gurobi_options"] = f"mipgap={GAP_REL} outlev=1"

    return ampl


def _set_time_limit(ampl, seconds):
    ampl.option["gurobi_options"] = (
        f"mipgap={GAP_REL} outlev=1 timelim={seconds}"
    )


def _solve_status(ampl):
    """Devuelve el texto de estado del solver (Optimal, Infeasible, etc.)."""
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
    """Convierte un DataFrame de amplpy (reset_index) en dict {(p,d,s): val}."""
    result = {}
    arr = df.values  # numpy array; columnas: p, d, s, value
    for row in arr:
        result[(row[0], int(row[1]), int(row[2]))] = row[3]
    return result


def _read_x(ampl):
    """Devuelve dict {(p,d,s): valor} para la variable x."""
    df = ampl.var["x"].get_values().to_pandas().reset_index()
    return _read_var3_df(df)


def _read_var3(ampl, var_name, index_set):
    """Lee una variable 3D (p in index_set, d, s) y devuelve dict."""
    if not index_set:
        return {}
    df = ampl.var[var_name].get_values().to_pandas().reset_index()
    return _read_var3_df(df)


def _day_assignments(x_vals, P):
    """Extrae asignacion de pacientes a dias desde la solucion de x."""
    Pd = {d: [] for d in DIAS}
    assigned = set()
    for (p, d, s), v in x_vals.items():
        if v > 0.5 and p not in assigned:
            Pd[d].append(p)
            assigned.add(p)
    return Pd


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEMA 1 — min F1
# ─────────────────────────────────────────────────────────────────────────────
def solve_p1(PB, PC, v_p, f_p):
    P = PB + PC
    print(f"\n{'─' * 60}")
    print(f"[P1] min overtime F1 | {len(PB)} criticos  {len(PC)} no-criticos")

    ampl = _build_ampl(PB, PC, v_p, f_p)
    # eps_f1 y eps_f2 inactivas (v1_bar=v2_bar=1e9 por defecto)
    ampl.eval("drop eps_f1; drop eps_f2;")
    ampl.eval("objective OvertimeTotal;")
    _set_time_limit(ampl, TIME_LIMIT_P1)

    t0 = time.time()
    ampl.solve()
    elapsed = time.time() - t0

    status = _solve_status(ampl)
    f1     = _obj_value(ampl, "OvertimeTotal")
    print(f"[P1] {status} | F1 = {f1:.1f} slots overtime | {elapsed:.1f}s")

    x_vals = _read_x(ampl)
    Pd     = _day_assignments(x_vals, P)

    aV_vals = {d: max(0.0, ampl.var["aV"][d].value()) for d in DIAS}
    aI_vals = {d: max(0.0, ampl.var["aI"][d].value()) for d in DIAS}

    return {
        "status":  status,
        "v1":      f1 if f1 is not None else float("inf"),
        "elapsed": elapsed,
        "Pd":      Pd,
        "aV":      aV_vals,
        "aI":      aI_vals,
        "ampl":    ampl,   # guardamos la instancia para extraer warm start
    }


# ─────────────────────────────────────────────────────────────────────────────
# DESCOMPOSICION DIARIA — Procedimiento 2 (warm start para P2)
# ─────────────────────────────────────────────────────────────────────────────
def _solve_day_p2(d, PB_d, PC_d, v_p, f_p, aV_fixed, aI_fixed):
    """Sub-problema de un dia para generar warm start de P2."""
    P_d = PB_d + PC_d
    if not P_d:
        return {"om": 0.0, "x": {}, "y": {}, "zB": {}, "zC": {}}

    ampl = _build_ampl(PB_d, PC_d, v_p, f_p)

    # Solo un dia
    ampl.set["D"] = [d]

    # Fijar overtime a valores optimos de P1
    ampl.var["aV"][d].fix(aV_fixed)
    ampl.var["aI"][d].fix(aI_fixed)

    ampl.eval("drop eps_f1; drop eps_f2;")
    ampl.eval("objective WaitTotal;")
    _set_time_limit(ampl, TIME_LIMIT_DAY)
    ampl.option["gurobi_options"] = (
        f"mipgap={GAP_REL} outlev=0 timelim={TIME_LIMIT_DAY}"
    )

    ampl.solve()

    status = _solve_status(ampl)
    if "infeasible" in status.lower():
        print(f"  [WARN] sub-problema dia {d} infactible — warm start parcial")
        return {"om": 0.0, "x": {}, "y": {}, "zB": {}, "zC": {}}

    om_val = max(0.0, ampl.var["om"][d].value())
    sol = {
        "om":  om_val,
        "x":   _read_x(ampl),
        "y":   _read_var3(ampl, "y",  PB_d),
        "zB":  _read_var3(ampl, "zB", PC_d),
        "zC":  _read_var3(ampl, "zC", PC_d),
    }
    return sol


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEMA 2 — min F2
# ─────────────────────────────────────────────────────────────────────────────
def solve_p2(PB, PC, v_p, f_p, r1):
    P   = PB + PC
    v1  = r1["v1"]
    Pd  = r1["Pd"]
    aV0 = r1["aV"]
    aI0 = r1["aI"]

    print(f"\n{'─' * 60}")
    print(f"[P2] min espera maxima diaria F2 | v1_bar = {v1:.1f} slots")
    print("[P2] Procedimiento 2: descomposicion por dia...")

    day_sols = {}
    for d in DIAS:
        PB_d = [p for p in Pd[d] if p in PB]
        PC_d = [p for p in Pd[d] if p in PC]
        s    = _solve_day_p2(d, PB_d, PC_d, v_p, f_p, aV0[d], aI0[d])
        day_sols[d] = s
        print(f"  Dia {d}: {len(Pd[d])} pacientes | omega = {s['om']:.1f} slots")

    # Problema P2 completo con warm start
    ampl = _build_ampl(PB, PC, v_p, f_p)
    ampl.eval("drop eps_f2;")          # solo eps_f1 activa
    ampl.param["v1_bar"] = v1
    ampl.eval("objective WaitTotal;")
    _set_time_limit(ampl, TIME_LIMIT_P2)

    # Warm start: inyectar solucion de sub-problemas diarios
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
    for d in DIAS:
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
    Pd2    = _day_assignments(x_vals, P)

    return {
        "status":  status,
        "v2":      f2 if f2 is not None else float("inf"),
        "elapsed": elapsed,
        "Pd":      Pd2,
        "aV":      {d: max(0.0, ampl.var["aV"][d].value()) for d in DIAS},
        "aI":      {d: max(0.0, ampl.var["aI"][d].value()) for d in DIAS},
        "om":      {d: max(0.0, ampl.var["om"][d].value())  for d in DIAS},
        "ampl":    ampl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEMA 3 — max F3
# ─────────────────────────────────────────────────────────────────────────────
def solve_p3(PB, PC, v_p, f_p, r1, r2):
    P  = PB + PC
    v1 = r1["v1"]
    v2 = r2["v2"]
    a2 = r2["ampl"]   # instancia P2 para extraer warm start

    print(f"\n{'─' * 60}")
    print(f"[P3] max sillones F3 | v1_bar = {v1:.1f}  v2_bar = {v2:.1f}")

    ampl = _build_ampl(PB, PC, v_p, f_p)
    ampl.param["v1_bar"] = v1
    ampl.param["v2_bar"] = v2
    ampl.eval("objective ChairsTotal;")
    _set_time_limit(ampl, TIME_LIMIT_P3)

    # Warm start desde solucion de P2
    for p in P:
        for d in DIAS:
            for s in SLOTS_VISITA:
                try:
                    ampl.var["x"][p, d, s].set_value(round(a2.var["x"][p, d, s].value() or 0))
                except Exception:
                    pass
    for p in PB:
        for d in DIAS:
            for s in SLOTS_INFUSION:
                try:
                    ampl.var["y"][p, d, s].set_value(round(a2.var["y"][p, d, s].value() or 0))
                except Exception:
                    pass
    for p in PC:
        for d in DIAS:
            for s in SLOTS_INFUSION:
                try:
                    ampl.var["zB"][p, d, s].set_value(round(a2.var["zB"][p, d, s].value() or 0))
                    ampl.var["zC"][p, d, s].set_value(round(a2.var["zC"][p, d, s].value() or 0))
                except Exception:
                    pass
    for d in DIAS:
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
        "aV":      {d: max(0.0, ampl.var["aV"][d].value()) for d in DIAS},
        "aI":      {d: max(0.0, ampl.var["aI"][d].value()) for d in DIAS},
        "om":      {d: max(0.0, ampl.var["om"][d].value())  for d in DIAS},
        "ampl":    ampl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCION DE AGENDA FINAL
# ─────────────────────────────────────────────────────────────────────────────
def _extract_agenda(ampl3, PB, PC, v_p):
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

        for d in DIAS:
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
def build_and_solve(df_input):
    PB, PC, v_p, f_p = _prepare_data(df_input)
    n = len(PB) + len(PC)
    print(f"\nPacientes totales: {n}  ({len(PB)} criticos / {len(PC)} no-criticos)")

    t_start = time.time()

    r1 = solve_p1(PB, PC, v_p, f_p)
    if r1["v1"] == float("inf"):
        print("[ERROR] P1 no encontro solucion factible.")
        return {"feasible": False}

    r2 = solve_p2(PB, PC, v_p, f_p, r1)
    if r2["v2"] == float("inf"):
        print("[ERROR] P2 no encontro solucion factible.")
        return {"feasible": False}

    r3 = solve_p3(PB, PC, v_p, f_p, r1, r2)

    t_total = time.time() - t_start

    df_agenda, complete = _extract_agenda(r3["ampl"], PB, PC, v_p)

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
def save_agenda(df, src_csv, frac_label):
    stem     = os.path.splitext(os.path.basename(src_csv))[0]
    out_path = os.path.join(OPTIMIZED_DIR, f"agenda_{stem}_frac{frac_label}_ampl.csv")
    df.sort_values(["Dia", "Slot_Inicio_Visita"]).to_csv(out_path, index=False)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# BUCLE PRINCIPAL — fracciones progresivas con criterio de aceptacion
# ─────────────────────────────────────────────────────────────────────────────
def run_progressive():
    # Activar licencia AMPL (incluye Gurobi como solver)
    modules.activate(AMPL_UUID)

    os.makedirs(OPTIMIZED_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    files = get_csv_files()
    if not files:
        print(f"No se encontraron CSV en {SIMULATED_DIR}.")
        return

    resumen = []

    for i, csv_path in enumerate(files):
        seed    = BASE_SEED + i
        df_full = pd.read_csv(csv_path)
        subsets = sample_prefixes(df_full, seed=seed)
        stem    = os.path.splitext(os.path.basename(csv_path))[0]

        prev_wait = None
        stop      = False

        for frac_label, _ in FRACCIONES:
            if stop:
                break

            log_path = os.path.join(LOGS_DIR, f"run_{stem}_frac{frac_label}.txt")
            df_part  = subsets[frac_label]

            with Tee(log_path):
                print("=" * 70)
                print(f"INSTANCIA : {stem}  [AMPL + Gurobi]")
                print(f"FRACCION  : {frac_label}  |  N = {len(df_part)}  |  seed = {seed}")
                print(f"FECHA     : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 70)

                res = build_and_solve(df_part)

                accepted = False
                reason   = ""

                if not res.get("feasible"):
                    reason = "infeasible_o_incoherente"
                    stop   = True
                else:
                    mw = res["mean_wait"]
                    if prev_wait is None:
                        accepted, reason = True, "primera_solucion_factible"
                    elif mw <= prev_wait:
                        accepted, reason = True, "factible_mejora_espera"
                    else:
                        reason = "factible_sin_mejora"
                        stop   = True

                print(f"\n{'─' * 70}")
                print(f"DECISION  : {'ACEPTADA' if accepted else 'RECHAZADA'}  ({reason})")

                out_path = None
                if accepted:
                    out_path  = save_agenda(res["results_df"], csv_path, frac_label)
                    prev_wait = res["mean_wait"]
                    print(f"Agenda guardada: {out_path}")

            resumen.append({
                "archivo":            os.path.basename(csv_path),
                "fraccion":           frac_label,
                "seed":               seed,
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
                "aceptada":           accepted,
                "motivo":             reason,
                "log":                log_path,
                "agenda":             out_path or "",
            })

    df_res   = pd.DataFrame(resumen)
    res_path = os.path.join(OPTIMIZED_DIR, "resumen_ejecucion_ampl.csv")
    df_res.to_csv(res_path, index=False)
    print(f"\nResumen global guardado en: {res_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_progressive()
