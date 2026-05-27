# Modelo MILP lexicografico para citacion en hospital de dia oncologico
# Implementacion fiel a Carello, Passacantando & Tanfani (EJOR, 2025)
# Secuencia: P1 (min overtime F1) -> P2 (min max-espera F2) -> P3 (max sillon F3)

import math
import os
import re
import sys
import time
from glob import glob

import numpy as np
import pandas as pd
from pulp import (
    LpMaximize,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
    value,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────────────────────
SIMULATED_DIR = os.path.join("Databases", "simulated-sets")
OPTIMIZED_DIR = os.path.join("Databases", "optimized-sets")
LOGS_DIR      = "logs"
BASE_SEED     = 20260427

DIAS = [1, 2, 3, 4, 5]

# Articulo: |SV|=48 (36 regulares 8:00-14:00 + 12 overtime), |SI|=66 (54 + 12)
SLOTS_VISITA   = range(1, 49)
SLOTS_INFUSION = range(1, 67)
NV = 36   # ultimo slot regular visitas  -> 14:00
NI = 54   # ultimo slot regular infusion -> 17:00

MAX_SALAS    = 6
MAX_SILLONES = 26
MAX_CAMAS    = 27

TIME_LIMIT_P1  = 600    # seg
TIME_LIMIT_P2  = 600
TIME_LIMIT_P3  = 1200
TIME_LIMIT_DAY = 120    # sub-problemas de descomposicion diaria
GAP_REL        = 0.05
SOLVER_MSG     = True

FRACCIONES = [
    ("1of3", 1 / 3),
    ("2of3", 2 / 3),
    ("3of3", 1.0),
]

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
    v_p  = dict(zip(df["ID_Paciente"], df["Visita_Slots"]))
    f_p  = dict(zip(df["ID_Paciente"], df["Infusion_Slots"]))
    crit = dict(zip(df["ID_Paciente"], df["Critico_Cama"]))
    PB   = [p for p in ids if crit[p] == 1]   # criticos  -> cama (y)
    PC   = [p for p in ids if crit[p] == 0]   # no-criticos -> sillon/cama (zC/zB)
    return PB, PC, v_p, f_p


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUES DE RESTRICCIONES
# ─────────────────────────────────────────────────────────────────────────────
def _core_constraints(prob, x, y, zB, zC, PB, PC, v_p, f_p, dias):
    """Restricciones (1)-(8) del articulo de Carello et al."""
    P = PB + PC

    # (1) Unicidad: cada paciente citado exactamente una vez en el horizonte
    for p in P:
        prob += lpSum(x[p][d][s] for d in dias for s in SLOTS_VISITA) == 1

    # (2) Capacidad salas de consulta (global, aprox. sin MCP por patologia)
    for d in dias:
        for s in SLOTS_VISITA:
            prob += lpSum(
                x[p][d][q]
                for p in P
                for q in range(max(1, s - v_p[p] + 1), s + 1)
            ) <= MAX_SALAS

    # (3) Mismo dia visita e infusion — pacientes criticos
    for p in PB:
        for d in dias:
            prob += (
                lpSum(x[p][d][s] for s in SLOTS_VISITA)
                == lpSum(y[p][d][s] for s in SLOTS_INFUSION)
            )

    # (4) Mismo dia visita e infusion — pacientes no-criticos
    for p in PC:
        for d in dias:
            prob += (
                lpSum(x[p][d][s] for s in SLOTS_VISITA)
                == lpSum(zB[p][d][s] + zC[p][d][s] for s in SLOTS_INFUSION)
            )

    # (5) Secuencialidad infusion tras visita — criticos
    for p in PB:
        for d in dias:
            prob += (
                lpSum(s * y[p][d][s] for s in SLOTS_INFUSION)
                >= lpSum((s + v_p[p]) * x[p][d][s] for s in SLOTS_VISITA)
            )

    # (6) Secuencialidad infusion tras visita — no-criticos
    for p in PC:
        for d in dias:
            prob += (
                lpSum(s * (zB[p][d][s] + zC[p][d][s]) for s in SLOTS_INFUSION)
                >= lpSum((s + v_p[p]) * x[p][d][s] for s in SLOTS_VISITA)
            )

    # (7) Capacidad sillones — exclusivamente no-criticos con zC
    for d in dias:
        for s in SLOTS_INFUSION:
            prob += lpSum(
                zC[p][d][q]
                for p in PC
                for q in range(max(1, s - f_p[p] + 1), s + 1)
            ) <= MAX_SILLONES

    # (8) Capacidad camas — criticos (y) + no-criticos en cama (zB)
    for d in dias:
        for s in SLOTS_INFUSION:
            prob += (
                lpSum(y[p][d][q]
                      for p in PB
                      for q in range(max(1, s - f_p[p] + 1), s + 1))
                + lpSum(zB[p][d][q]
                        for p in PC
                        for q in range(max(1, s - f_p[p] + 1), s + 1))
            ) <= MAX_CAMAS

    return prob


def _overtime_constraints(prob, x, y, zB, zC, aV, aI, PB, PC, v_p, f_p, dias):
    """Restricciones (9a-9c): overtime de visitas e infusiones."""
    # (9a) Overtime visitas — todos los pacientes
    for d in dias:
        for p in PB + PC:
            prob += (
                aV[d] >= lpSum((s + v_p[p] - 1) * x[p][d][s] for s in SLOTS_VISITA) - NV
            )

    # (9b) Overtime infusiones — criticos
    for d in dias:
        for p in PB:
            prob += (
                aI[d] >= lpSum((s + f_p[p] - 1) * y[p][d][s] for s in SLOTS_INFUSION) - NI
            )

    # (9c) Overtime infusiones — no-criticos
    for d in dias:
        for p in PC:
            prob += (
                aI[d] >= lpSum(
                    (s + f_p[p] - 1) * (zB[p][d][s] + zC[p][d][s])
                    for s in SLOTS_INFUSION
                ) - NI
            )

    return prob


def _wait_constraints(prob, x, y, zB, zC, om, PB, PC, v_p, dias):
    """Restriccion (10): omega_d >= espera de cada paciente el dia d."""
    for d in dias:
        for p in PB:
            prob += (
                om[d]
                >= lpSum(s * y[p][d][s] for s in SLOTS_INFUSION)
                - lpSum((s + v_p[p]) * x[p][d][s] for s in SLOTS_VISITA)
            )
        for p in PC:
            prob += (
                om[d]
                >= lpSum(s * (zB[p][d][s] + zC[p][d][s]) for s in SLOTS_INFUSION)
                - lpSum((s + v_p[p]) * x[p][d][s] for s in SLOTS_VISITA)
            )

    return prob


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEMA (1) — min F1 = Sigma_d (alphaV_d + alphaI_d)
# ─────────────────────────────────────────────────────────────────────────────
def solve_p1(PB, PC, v_p, f_p):
    P = PB + PC
    print(f"\n{'─' * 60}")
    print(f"[P1] min overtime F1 | {len(PB)} criticos  {len(PC)} no-criticos")

    x  = LpVariable.dicts("x",  (P,  DIAS, SLOTS_VISITA),   cat="Binary")
    y  = LpVariable.dicts("y",  (PB, DIAS, SLOTS_INFUSION), cat="Binary")
    zB = LpVariable.dicts("zB", (PC, DIAS, SLOTS_INFUSION), cat="Binary")
    zC = LpVariable.dicts("zC", (PC, DIAS, SLOTS_INFUSION), cat="Binary")
    aV = LpVariable.dicts("aV", DIAS, lowBound=0)
    aI = LpVariable.dicts("aI", DIAS, lowBound=0)

    prob = LpProblem("P1_Overtime", LpMinimize)
    prob += lpSum(aV[d] + aI[d] for d in DIAS)

    _core_constraints(prob, x, y, zB, zC, PB, PC, v_p, f_p, DIAS)
    _overtime_constraints(prob, x, y, zB, zC, aV, aI, PB, PC, v_p, f_p, DIAS)

    t0 = time.time()
    PULP_CBC_CMD(msg=SOLVER_MSG, timeLimit=TIME_LIMIT_P1, gapRel=GAP_REL).solve(prob)
    elapsed = time.time() - t0

    status = LpStatus[prob.status]
    f1     = value(prob.objective)
    print(f"[P1] {status} | F1 = {f1:.1f} slots overtime | {elapsed:.1f}s")

    # Asignacion de pacientes a dias segun solucion de P1
    Pd = {d: [] for d in DIAS}
    for p in P:
        for d in DIAS:
            if any((value(x[p][d][s]) or 0) > 0.5 for s in SLOTS_VISITA):
                Pd[d].append(p)
                break

    aV_vals = {d: max(0.0, value(aV[d]) or 0.0) for d in DIAS}
    aI_vals = {d: max(0.0, value(aI[d]) or 0.0) for d in DIAS}

    return {
        "status":  status,
        "v1":      f1 if f1 is not None else float("inf"),
        "elapsed": elapsed,
        "Pd":      Pd,
        "aV":      aV_vals,
        "aI":      aI_vals,
        "vars":    {"x": x, "y": y, "zB": zB, "zC": zC},
    }


# ─────────────────────────────────────────────────────────────────────────────
# DESCOMPOSICION DIARIA — Procedimiento 2 del articulo (warm start para P2)
# ─────────────────────────────────────────────────────────────────────────────
def _solve_day_p2(d, PB_d, PC_d, v_p, f_p, aV_fixed, aI_fixed):
    """
    Resuelve P2 para un unico dia d con pacientes fijos y overtime fijo (de P1).
    Implementa el Procedimiento 2 del articulo para generar el warm start de P2.
    """
    P_d = PB_d + PC_d
    if not P_d:
        return {"x": {}, "y": {}, "zB": {}, "zC": {}, "om": 0.0}

    prob = LpProblem(f"P2_dia{d}", LpMinimize)

    x_d  = LpVariable.dicts("x",  (P_d,  [d], SLOTS_VISITA),   cat="Binary")
    y_d  = LpVariable.dicts("y",  (PB_d, [d], SLOTS_INFUSION), cat="Binary") if PB_d else {}
    zB_d = LpVariable.dicts("zB", (PC_d, [d], SLOTS_INFUSION), cat="Binary") if PC_d else {}
    zC_d = LpVariable.dicts("zC", (PC_d, [d], SLOTS_INFUSION), cat="Binary") if PC_d else {}
    om_d = LpVariable("om", lowBound=0)

    prob += om_d

    # Unicidad (cada paciente en P_d ya esta asignado a este dia)
    for p in P_d:
        prob += lpSum(x_d[p][d][s] for s in SLOTS_VISITA) == 1

    # Capacidad salas
    for s in SLOTS_VISITA:
        prob += lpSum(
            x_d[p][d][q]
            for p in P_d
            for q in range(max(1, s - v_p[p] + 1), s + 1)
        ) <= MAX_SALAS

    # Mismo dia (trivial, dia unico) + secuencialidad
    for p in PB_d:
        prob += (lpSum(x_d[p][d][s] for s in SLOTS_VISITA)
                 == lpSum(y_d[p][d][s] for s in SLOTS_INFUSION))
        prob += (lpSum(s * y_d[p][d][s] for s in SLOTS_INFUSION)
                 >= lpSum((s + v_p[p]) * x_d[p][d][s] for s in SLOTS_VISITA))
    for p in PC_d:
        prob += (lpSum(x_d[p][d][s] for s in SLOTS_VISITA)
                 == lpSum(zB_d[p][d][s] + zC_d[p][d][s] for s in SLOTS_INFUSION))
        prob += (lpSum(s * (zB_d[p][d][s] + zC_d[p][d][s]) for s in SLOTS_INFUSION)
                 >= lpSum((s + v_p[p]) * x_d[p][d][s] for s in SLOTS_VISITA))

    # Capacidad sillones y camas
    for s in SLOTS_INFUSION:
        if PC_d:
            prob += lpSum(
                zC_d[p][d][q]
                for p in PC_d
                for q in range(max(1, s - f_p[p] + 1), s + 1)
            ) <= MAX_SILLONES
        prob += (
            lpSum(y_d[p][d][q]
                  for p in PB_d
                  for q in range(max(1, s - f_p[p] + 1), s + 1))
            + lpSum(zB_d[p][d][q]
                    for p in PC_d
                    for q in range(max(1, s - f_p[p] + 1), s + 1))
        ) <= MAX_CAMAS

    # Overtime fijado a los valores optimos de P1
    aV_var = LpVariable("aV", lowBound=aV_fixed, upBound=aV_fixed)
    aI_var = LpVariable("aI", lowBound=aI_fixed, upBound=aI_fixed)
    for p in P_d:
        prob += aV_var >= lpSum((s + v_p[p] - 1) * x_d[p][d][s] for s in SLOTS_VISITA) - NV
    for p in PB_d:
        prob += aI_var >= lpSum((s + f_p[p] - 1) * y_d[p][d][s] for s in SLOTS_INFUSION) - NI
    for p in PC_d:
        prob += aI_var >= lpSum(
            (s + f_p[p] - 1) * (zB_d[p][d][s] + zC_d[p][d][s]) for s in SLOTS_INFUSION
        ) - NI

    # Restriccion (10) para este dia
    for p in PB_d:
        prob += (om_d >= lpSum(s * y_d[p][d][s] for s in SLOTS_INFUSION)
                 - lpSum((s + v_p[p]) * x_d[p][d][s] for s in SLOTS_VISITA))
    for p in PC_d:
        prob += (om_d >= lpSum(s * (zB_d[p][d][s] + zC_d[p][d][s]) for s in SLOTS_INFUSION)
                 - lpSum((s + v_p[p]) * x_d[p][d][s] for s in SLOTS_VISITA))

    PULP_CBC_CMD(msg=False, timeLimit=TIME_LIMIT_DAY, gapRel=GAP_REL).solve(prob)

    if LpStatus[prob.status] == "Infeasible":
        print(f"  [WARN] sub-problema dia {d} infactible — warm start parcial")
        return {"x": {}, "y": {}, "zB": {}, "zC": {}, "om": 0.0}

    sol = {"x": {}, "y": {}, "zB": {}, "zC": {}, "om": max(0.0, value(om_d) or 0.0)}
    for p in P_d:
        for s in SLOTS_VISITA:
            v = value(x_d[p][d][s])
            if v is not None:
                sol["x"][(p, d, s)] = round(v)
    for p in PB_d:
        for s in SLOTS_INFUSION:
            v = value(y_d[p][d][s])
            if v is not None:
                sol["y"][(p, d, s)] = round(v)
    for p in PC_d:
        for s in SLOTS_INFUSION:
            vB = value(zB_d[p][d][s])
            vC = value(zC_d[p][d][s])
            if vB is not None:
                sol["zB"][(p, d, s)] = round(vB)
            if vC is not None:
                sol["zC"][(p, d, s)] = round(vC)

    return sol


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEMA (2) — min F2 = Sigma_d omega_d,  s.t.  F1 <= v_bar1
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

    # Construccion del problema P2 completo
    x  = LpVariable.dicts("x",  (P,  DIAS, SLOTS_VISITA),   cat="Binary")
    y  = LpVariable.dicts("y",  (PB, DIAS, SLOTS_INFUSION), cat="Binary")
    zB = LpVariable.dicts("zB", (PC, DIAS, SLOTS_INFUSION), cat="Binary")
    zC = LpVariable.dicts("zC", (PC, DIAS, SLOTS_INFUSION), cat="Binary")
    aV = LpVariable.dicts("aV", DIAS, lowBound=0)
    aI = LpVariable.dicts("aI", DIAS, lowBound=0)
    om = LpVariable.dicts("om", DIAS, lowBound=0)

    prob = LpProblem("P2_Wait", LpMinimize)
    prob += lpSum(om[d] for d in DIAS)

    _core_constraints(prob, x, y, zB, zC, PB, PC, v_p, f_p, DIAS)
    _overtime_constraints(prob, x, y, zB, zC, aV, aI, PB, PC, v_p, f_p, DIAS)
    _wait_constraints(prob, x, y, zB, zC, om, PB, PC, v_p, DIAS)
    prob += lpSum(aV[d] + aI[d] for d in DIAS) <= v1

    # Warm start: inicializar todo a 0, sobrescribir con solucion descompuesta
    for p in P:
        for d in DIAS:
            for s in SLOTS_VISITA:
                x[p][d][s].setInitialValue(0)
    for p in PB:
        for d in DIAS:
            for s in SLOTS_INFUSION:
                y[p][d][s].setInitialValue(0)
    for p in PC:
        for d in DIAS:
            for s in SLOTS_INFUSION:
                zB[p][d][s].setInitialValue(0)
                zC[p][d][s].setInitialValue(0)
    for d in DIAS:
        aV[d].setInitialValue(aV0[d])
        aI[d].setInitialValue(aI0[d])
        om[d].setInitialValue(day_sols[d]["om"])
    for d, sol in day_sols.items():
        for (p, dd, s), v in sol["x"].items():
            x[p][dd][s].setInitialValue(v)
        for (p, dd, s), v in sol["y"].items():
            y[p][dd][s].setInitialValue(v)
        for (p, dd, s), v in sol["zB"].items():
            zB[p][dd][s].setInitialValue(v)
        for (p, dd, s), v in sol["zC"].items():
            zC[p][dd][s].setInitialValue(v)

    print("[P2] Resolviendo problema completo con warm start...")
    t0 = time.time()
    PULP_CBC_CMD(msg=SOLVER_MSG, timeLimit=TIME_LIMIT_P2, gapRel=GAP_REL,
                 warmStart=True).solve(prob)
    elapsed = time.time() - t0

    status = LpStatus[prob.status]
    f2     = value(prob.objective)
    print(f"[P2] {status} | F2 = {f2:.1f} slots | {elapsed:.1f}s")

    Pd2 = {d: [] for d in DIAS}
    for p in P:
        for d in DIAS:
            if any((value(x[p][d][s]) or 0) > 0.5 for s in SLOTS_VISITA):
                Pd2[d].append(p)
                break

    return {
        "status":  status,
        "v2":      f2 if f2 is not None else float("inf"),
        "elapsed": elapsed,
        "Pd":      Pd2,
        "aV":      {d: max(0.0, value(aV[d]) or 0.0) for d in DIAS},
        "aI":      {d: max(0.0, value(aI[d]) or 0.0) for d in DIAS},
        "om":      {d: max(0.0, value(om[d])  or 0.0) for d in DIAS},
        "vars":    {"x": x, "y": y, "zB": zB, "zC": zC},
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEMA (3) — max F3 = Sigma zC,  s.t.  F1 <= v_bar1,  F2 <= v_bar2
# ─────────────────────────────────────────────────────────────────────────────
def solve_p3(PB, PC, v_p, f_p, r1, r2):
    P   = PB + PC
    v1  = r1["v1"]
    v2  = r2["v2"]
    p2v = r2["vars"]

    print(f"\n{'─' * 60}")
    print(f"[P3] max sillones F3 | v1_bar = {v1:.1f}  v2_bar = {v2:.1f}")

    x  = LpVariable.dicts("x",  (P,  DIAS, SLOTS_VISITA),   cat="Binary")
    y  = LpVariable.dicts("y",  (PB, DIAS, SLOTS_INFUSION), cat="Binary")
    zB = LpVariable.dicts("zB", (PC, DIAS, SLOTS_INFUSION), cat="Binary")
    zC = LpVariable.dicts("zC", (PC, DIAS, SLOTS_INFUSION), cat="Binary")
    aV = LpVariable.dicts("aV", DIAS, lowBound=0)
    aI = LpVariable.dicts("aI", DIAS, lowBound=0)
    om = LpVariable.dicts("om", DIAS, lowBound=0)

    prob = LpProblem("P3_Chairs", LpMaximize)
    prob += lpSum(zC[p][d][s] for p in PC for d in DIAS for s in SLOTS_INFUSION)

    _core_constraints(prob, x, y, zB, zC, PB, PC, v_p, f_p, DIAS)
    _overtime_constraints(prob, x, y, zB, zC, aV, aI, PB, PC, v_p, f_p, DIAS)
    _wait_constraints(prob, x, y, zB, zC, om, PB, PC, v_p, DIAS)
    prob += lpSum(aV[d] + aI[d] for d in DIAS) <= v1
    prob += lpSum(om[d] for d in DIAS) <= v2

    # Warm start desde solucion de P2
    for p in P:
        for d in DIAS:
            for s in SLOTS_VISITA:
                vi = value(p2v["x"][p][d][s])
                x[p][d][s].setInitialValue(round(vi) if vi is not None else 0)
    for p in PB:
        for d in DIAS:
            for s in SLOTS_INFUSION:
                vi = value(p2v["y"][p][d][s])
                y[p][d][s].setInitialValue(round(vi) if vi is not None else 0)
    for p in PC:
        for d in DIAS:
            for s in SLOTS_INFUSION:
                vB = value(p2v["zB"][p][d][s])
                vC = value(p2v["zC"][p][d][s])
                zB[p][d][s].setInitialValue(round(vB) if vB is not None else 0)
                zC[p][d][s].setInitialValue(round(vC) if vC is not None else 0)
    for d in DIAS:
        aV[d].setInitialValue(r2["aV"][d])
        aI[d].setInitialValue(r2["aI"][d])
        om[d].setInitialValue(r2["om"][d])

    t0 = time.time()
    PULP_CBC_CMD(msg=SOLVER_MSG, timeLimit=TIME_LIMIT_P3, gapRel=GAP_REL,
                 warmStart=True).solve(prob)
    elapsed = time.time() - t0

    status = LpStatus[prob.status]
    f3     = value(prob.objective)
    print(f"[P3] {status} | F3 = {f3:.0f} pacientes en sillon | {elapsed:.1f}s")

    return {
        "status":  status,
        "f3":      f3 if f3 is not None else 0.0,
        "elapsed": elapsed,
        "aV":      {d: max(0.0, value(aV[d]) or 0.0) for d in DIAS},
        "aI":      {d: max(0.0, value(aI[d]) or 0.0) for d in DIAS},
        "om":      {d: max(0.0, value(om[d])  or 0.0) for d in DIAS},
        "vars":    {"x": x, "y": y, "zB": zB, "zC": zC},
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCION DE AGENDA FINAL
# ─────────────────────────────────────────────────────────────────────────────
def _extract_agenda(x, y, zB, zC, PB, PC, v_p):
    rows     = []
    complete = True

    for p in PB + PC:
        fila    = {"ID_Paciente": p, "Critico": 1 if p in PB else 0}
        found_v = False
        found_i = False

        for d in DIAS:
            for s in SLOTS_VISITA:
                if (value(x[p][d][s]) or 0) > 0.5:
                    fila["Dia"]                = d
                    fila["Slot_Inicio_Visita"] = s
                    found_v = True

                    if p in PB:
                        for si in SLOTS_INFUSION:
                            if (value(y[p][d][si]) or 0) > 0.5:
                                fila["Slot_Inicio_Infusion"] = si
                                fila["Recurso_Infusion"]     = "cama"
                                found_i = True
                                break
                    else:
                        for si in SLOTS_INFUSION:
                            if (value(zC[p][d][si]) or 0) > 0.5:
                                fila["Slot_Inicio_Infusion"] = si
                                fila["Recurso_Infusion"]     = "sillon"
                                found_i = True
                                break
                            if (value(zB[p][d][si]) or 0) > 0.5:
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

    v3 = r3["vars"]
    df_agenda, complete = _extract_agenda(
        v3["x"], v3["y"], v3["zB"], v3["zC"], PB, PC, v_p
    )

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
    out_path = os.path.join(OPTIMIZED_DIR, f"agenda_{stem}_frac{frac_label}.csv")
    df.sort_values(["Dia", "Slot_Inicio_Visita"]).to_csv(out_path, index=False)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# BUCLE PRINCIPAL — fracciones progresivas con criterio de aceptacion
# ─────────────────────────────────────────────────────────────────────────────
def run_progressive():
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
                print(f"INSTANCIA : {stem}")
                print(f"FRACCION  : {frac_label}  |  N = {len(df_part)}  |  seed = {seed}")
                print(f"FECHA     : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 70)

                res = build_and_solve(df_part)

                # Criterio de aceptacion
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
    res_path = os.path.join(OPTIMIZED_DIR, "resumen_ejecucion.csv")
    df_res.to_csv(res_path, index=False)
    print(f"\nResumen global guardado en: {res_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_progressive()
