# Modelo MILP lexicografico para citacion en hospital de dia oncologico
# Datos reales: instancias .dat del articulo (San Martino Hospital, Genova)
# Carello, Passacantando & Tanfani (EJOR, 2025)
# Secuencia: P1 (min overtime F1) -> P2 (min max-espera F2) -> P3 (max sillon F3)
# Solver: Gurobi via amplpy
#
# VARIANTE MCP CRUDO — el MCP se lee DIRECTAMENTE del fichero .dat (sin re-balanceo).
# El objetivo es reproducir la problematica documentada en:
#   mds-to-word/tfg-agregados-completo.md (§8.3–8.6, §12)
#   mds-to-word/analisis-barrido-51-instancias.md
# Con el MCP crudo se espera:
#   - Deficit estructural en ~15 semanas de 5 dias (p.ej. HE en istanza03: 46 slots)
#   - istanza31 infactible (UR con 1 sala para 51 pacientes, techo |SV|=48)
#   - Overtime forzado donde el articulo da 0 (principio del palomar)
#   - Coincidencia numerica en istanza11/17/18 debida al MCP crudo, no a festivos
# Comparar con model-solver-real.py (Opcion C) para ver el efecto del re-balanceo.

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
OPTIMIZED_DIR = os.path.join("Databases", "optimized-sets-real-crudo")   # <- diferente
LOGS_DIR      = "logs-real-crudo"                                         # <- diferente

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

EXCLUDED_INSTANCES = {"istanza14"}

# ─────────────────────────────────────────────────────────────────────────────
# MODELO AMPL
# ─────────────────────────────────────────────────────────────────────────────
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
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    m = re.search(r"param\s+giornilav\s*:=\s*(\d+)", text)
    giornilav = int(m.group(1))

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

    w = {}
    w_block = re.search(r"param\s+w\s*:=(.*?);", text, re.DOTALL)
    if w_block:
        for entry in re.finditer(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s+1", w_block.group(1)):
            r, k, d = int(entry.group(1)), int(entry.group(2)), int(entry.group(3))
            w[(r, k, d)] = 1

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


# ─────────────────────────────────────────────────────────────────────────────
# MCP CRUDO — nsalas leido directamente del .dat
# ─────────────────────────────────────────────────────────────────────────────
# Para cada dia laborable (lab = 1..giornilav) se suma el numero de salas que
# el .dat asigna a cada patologia k en el dia NATURAL correspondiente (c_map[lab]).
# En semanas con festivo, el dia festivo no tiene ningún dia laborable que le
# apunte -> sus bloques se pierden (mecanismo del articulo), pero el reparto de
# los dias restantes es exactamente el del .dat (sin re-balanceo).
# Esto produce los deficits estructurales y la infactibilidad documentados en
# tfg-agregados-completo.md §8.4–8.6.
def prepare_instance(dat):
    g    = dat["giornilav"]
    dias = list(range(1, g + 1))
    c_map = dat["c_map"] or {i: i for i in dias}

    # nsalas[(k, lab)] = nº salas de patologia k el dia laborable lab
    nsalas = {(k, lab): 0 for k in range(1, 8) for lab in range(1, g + 1)}
    for lab in range(1, g + 1):
        d_nat = c_map[lab]
        for r in range(1, 7):
            for k in range(1, 8):
                if dat["w"].get((r, k, d_nat), 0):
                    nsalas[(k, lab)] += 1

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


def _print_deficit_info(dias, PB, PC, v_p, alpha_p, nsalas):
    """Log informativo del deficit estructural con el MCP crudo."""
    P = PB + PC
    salas_k = {k: sum(nsalas.get((k, d), 0) for d in dias) for k in range(1, 8)}
    dem_k   = {k: 0 for k in range(1, 8)}
    for p in P:
        dem_k[alpha_p[p]] += v_p[p]
    pat_names = {1: "HE", 2: "GI", 3: "UR", 4: "GY", 5: "BR", 6: "OT", 7: "LU"}
    for k in range(1, 8):
        deficit = dem_k[k] - NV * salas_k[k]
        if deficit > 0:
            print(f"  [DEFICIT MCP CRUDO] {pat_names[k]}: dem={dem_k[k]} "
                  f"salas={salas_k[k]} cap_regular={NV*salas_k[k]} "
                  f"DEFICIT={deficit} slots -> overtime FORZADO por palomar")
        elif salas_k[k] == 0 and dem_k[k] > 0:
            print(f"  [INFACTIBLE MCP CRUDO] {pat_names[k]}: dem={dem_k[k]} "
                  f"salas=0 -> SIN SALA para esta patologia (infactible)")


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

    ampl.param["NV"]    = NV
    ampl.param["NI"]    = NI
    ampl.param["RSILL"] = MAX_SILLONES
    ampl.param["RCAMA"] = MAX_CAMAS

    ampl.param["vp"]    = {p: v_p[p]    for p in P}
    ampl.param["fp"]    = {p: f_p[p]    for p in P}
    ampl.param["alpha"] = {p: alpha_p[p] for p in P}

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
# LOWER BOUND LB1 — Procedimiento 1 del articulo (§4.1), con correccion
# ─────────────────────────────────────────────────────────────────────────────
def compute_lb1(dias, PB, PC, v_p, f_p, alpha_p, nsalas):
    """Calcula la cota inferior VALIDA del overtime F1 (LB1 = LP reforzado).
    Identica a model-solver-real.py: lb_conteo solo como cross-check."""
    P = PB + PC

    salas_k = {k: sum(nsalas.get((k, d), 0) for d in dias) for k in range(1, 8)}
    dem_k   = {k: 0 for k in range(1, 8)}
    for p in P:
        dem_k[alpha_p[p]] += v_p[p]
    K_bar = [k for k in range(1, 8) if NV * salas_k[k] < dem_k[k]]

    if not K_bar:
        return 0.0, {"K_bar": [], "n_pbar": 0, "lb_conteo": 0.0, "lb_lp": 0.0}

    lb_conteo = sum(max(0, dem_k[k] - NV * salas_k[k]) for k in K_bar)

    P_bar  = [p for p in P if alpha_p[p] in K_bar]
    PB_bar = [p for p in P_bar if p in PB]
    PC_bar = [p for p in P_bar if p in PC]

    ampl = _build_ampl(dias, PB_bar, PC_bar, v_p, f_p, alpha_p, nsalas)
    ampl.eval("drop eps_f1; drop eps_f2;")
    ampl.param["RSILL"] = 10 ** 6
    ampl.param["RCAMA"] = 10 ** 6
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

    lb1, lb1_info = compute_lb1(dias, PB, PC, v_p, f_p, alpha_p, w_mcp)
    _xchk = " [!] conteo>LB1: cota de conteo NO valida, se ignora" \
        if lb1_info['lb_conteo'] > lb1 + 1e-6 else ""
    print(f"[P1] LB1 = {lb1:.1f} (LP reforzado)  "
          f"[conteo_xcheck={lb1_info['lb_conteo']:.1f}] | deficit={lb1_info['K_bar']} "
          f"{lb1_info['n_pbar']} pac{_xchk}")

    ampl = _build_ampl(dias, PB, PC, v_p, f_p, alpha_p, w_mcp)
    ampl.eval("drop eps_f1; drop eps_f2;")
    ampl.eval("objective OvertimeTotal;")
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
# BUCLE PRINCIPAL
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
    print("MODO: MCP CRUDO (sin re-balanceo — nsalas leido directamente del .dat)")
    print("  Esperado: deficit estructural en ~15 semanas; istanza31 infactible.")

    resumen = []

    for dat_path in dat_files:
        stem = os.path.splitext(os.path.basename(dat_path))[0]

        if stem in EXCLUDED_INSTANCES:
            print(f"[SKIP] {stem}: descartada (no figura en la Tabla 7 del articulo).")
            continue

        log_path = os.path.join(LOGS_DIR, f"run_{stem}.txt")

        with Tee(log_path):
            print("=" * 70)
            print(f"INSTANCIA : {stem}  [MCP CRUDO — sin re-balanceo]")
            print(f"FICHERO   : {dat_path}")
            print(f"FECHA     : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)

            dat = parse_dat(dat_path)
            dias, PB, PC, v_p, f_p, alpha_p, w_mcp = prepare_instance(dat)

            print(f"giornilav = {dat['giornilav']}  |  ptot = {len(dat['patients'])}")
            _print_deficit_info(dias, PB, PC, v_p, alpha_p, w_mcp)

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
    res_path = os.path.join(OPTIMIZED_DIR, "resumen_ejecucion_real_crudo.csv")
    df_res.to_csv(res_path, index=False)
    print(f"\nResumen global guardado en: {res_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_all()
