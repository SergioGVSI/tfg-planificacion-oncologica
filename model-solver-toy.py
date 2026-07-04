# Toy problem para estudiar y verificar el modelo MILP del solver de datos reales.
#
# Objetivo: una instancia INVENTADA, pequeña y con valores calculables a mano, que
# ejercite todas las restricciones (1)-(10) del articulo y los tres problemas P1->P2->P3.
# Sirve para:
#   - verificar que el solver da el F1/F2/F3 que se predice analiticamente,
#   - iterar mejoras (p.ej. el lower bound LB1) en SEGUNDOS en vez de minutos.
#
# El modelo AMPL es el mismo que model-solver-real.py pero con dimensiones reducidas
# (slots pequenos) para poder razonar la solucion optima con lapiz y papel.

import time

from amplpy import AMPL, modules

AMPL_UUID = ""

# ─────────────────────────────────────────────────────────────────────────────
# DIMENSIONES REDUCIDAS DEL TOY (no son las del articulo; elegidas para calcular a mano)
# ─────────────────────────────────────────────────────────────────────────────
# Visitas: 4 slots regulares + 2 de overtime = 6 slots totales
NV = 4
SLOTS_VISITA = list(range(1, 7))      # |SV| = 6
# Infusiones: 6 slots regulares + 3 de overtime = 9 slots totales
NI = 6
SLOTS_INFUSION = list(range(1, 10))   # |SI| = 9

DIAS = [1]            # un solo dia, para razonar la agenda facilmente
K = [1, 2]            # dos patologias
RSILL = 2             # 2 sillones (holgura para que la infusion NO genere overtime)
RCAMA = 1             # 1 cama

# MCP del toy: nsalas[k,d] = nº de salas de la patologia k en el dia d.
# Pat 1 -> 1 sala el dia 1.  Pat 2 -> 1 sala el dia 1.
NSALAS = {(1, 1): 1, (2, 1): 1}

# ─────────────────────────────────────────────────────────────────────────────
# PACIENTES DEL TOY
# ─────────────────────────────────────────────────────────────────────────────
# Cada paciente: id, patologia (alpha), duracion visita (v), duracion infusion (f),
# critico (1 = requiere cama, 0 = prefiere sillon).
#
# Diseno para FORZAR overtime de visita conocido:
#   - Pat 1 tiene 1 sala => capacidad regular de visita = NV = 4 slots.
#   - Metemos 5 pacientes de pat 1, cada uno con visita de 1 slot.
#   - Como mucho 4 caben en los slots regulares (1..4); el 5º empieza en el slot 5
#     (overtime). Su visita termina en el slot 5, y NV=4, asi que genera
#     aV >= (5 + 1 - 1) - 4 = 1 slot de overtime de VISITA.  (restriccion 9a)
#   - Pat 2 tiene 2 pacientes de visita 1 slot: caben de sobra en su sala (sin overtime).
TOY_PATIENTS = [
    # id, alpha, v, f, critico
    (1, 1, 1, 2, 0),   # pat1 no-critico (sillon)
    (2, 1, 1, 2, 0),   # pat1 no-critico
    (3, 1, 1, 2, 0),   # pat1 no-critico
    (4, 1, 1, 2, 1),   # pat1 critico (cama)
    (5, 1, 1, 2, 0),   # pat1 no-critico -> este empuja a overtime de visita
    (6, 2, 1, 2, 0),   # pat2 no-critico
    (7, 2, 1, 2, 1),   # pat2 critico (cama)
]

# ─────────────────────────────────────────────────────────────────────────────
# MODELO AMPL (identico en estructura al de model-solver-real.py)
# ─────────────────────────────────────────────────────────────────────────────
AMPL_MODEL = """
set P;          # pacientes
set PB;         # criticos (cama)
set PC;         # no-criticos (sillon o cama)
set D;          # dias
set SV;         # slots de visita
set SI;         # slots de infusion
set K;          # patologias

param vp{P}      integer >= 1;
param fp{P}      integer >= 1;
param alpha{P}   integer >= 1;
param nsalas{K, D} integer >= 0;
param NV         integer;
param NI         integer;
param RSILL      integer;
param RCAMA      integer;

param v1_bar default 1e9;
param v2_bar default 1e9;

var x{P, D, SV}  binary;
var y{PB, D, SI} binary;
var zB{PC, D, SI} binary;
var zC{PC, D, SI} binary;
var aV{D} >= 0;
var aI{D} >= 0;
var om{D} >= 0;

minimize OvertimeTotal: sum{d in D} (aV[d] + aI[d]);
minimize WaitTotal:     sum{d in D} om[d];
maximize ChairsTotal:   sum{p in PC, d in D, s in SI} zC[p,d,s];

# (1) Unicidad
subject to unicidad{p in P}:
    sum{d in D, s in SV} x[p,d,s] = 1;

# (2) Capacidad salas por patologia (MCP)
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

# (7) Capacidad sillones
subject to cap_sillones{d in D, s in SI}:
    sum{p in PC, q in SI: q >= max(1, s-fp[p]+1) and q <= s} zC[p,d,q] <= RSILL;

# (8) Capacidad camas
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

subject to eps_f1: sum{d in D} (aV[d] + aI[d]) <= v1_bar;
subject to eps_f2: sum{d in D} om[d] <= v2_bar;
"""


def _build():
    PB = [p[0] for p in TOY_PATIENTS if p[4] == 1]
    PC = [p[0] for p in TOY_PATIENTS if p[4] == 0]
    P  = [p[0] for p in TOY_PATIENTS]
    v_p   = {p[0]: p[2] for p in TOY_PATIENTS}
    f_p   = {p[0]: p[3] for p in TOY_PATIENTS}
    alpha = {p[0]: p[1] for p in TOY_PATIENTS}

    ampl = AMPL()
    ampl.eval(AMPL_MODEL)

    ampl.set["P"]  = P
    ampl.set["PB"] = PB
    ampl.set["PC"] = PC
    ampl.set["D"]  = DIAS
    ampl.set["SV"] = SLOTS_VISITA
    ampl.set["SI"] = SLOTS_INFUSION
    ampl.set["K"]  = K

    ampl.param["NV"]    = NV
    ampl.param["NI"]    = NI
    ampl.param["RSILL"] = RSILL
    ampl.param["RCAMA"] = RCAMA
    ampl.param["vp"]    = v_p
    ampl.param["fp"]    = f_p
    ampl.param["alpha"] = alpha
    ampl.param["nsalas"] = {(k, d): NSALAS.get((k, d), 0) for k in K for d in DIAS}

    ampl.option["solver"] = "gurobi"
    ampl.option["gurobi_options"] = "outlev=0"
    return ampl, PB, PC


def _status(ampl):
    return ampl.get_value("solve_result")


def _obj(ampl, name):
    try:
        return ampl.get_objective(name).value()
    except Exception:
        return None


def _agenda(ampl, PB, PC):
    """Imprime la agenda final (dia, slot visita, slot infusion, recurso, espera)."""
    PB_set = set(PB)
    print("\n  Agenda (paciente: dia, slot_visita, slot_infusion, recurso, espera):")
    for p in PB + PC:
        for d in DIAS:
            for s in SLOTS_VISITA:
                if (ampl.var["x"][p, d, s].value() or 0) > 0.5:
                    vis = s
                    inf, rec = None, None
                    if p in PB_set:
                        for si in SLOTS_INFUSION:
                            if (ampl.var["y"][p, d, si].value() or 0) > 0.5:
                                inf, rec = si, "cama"; break
                    else:
                        for si in SLOTS_INFUSION:
                            if (ampl.var["zC"][p, d, si].value() or 0) > 0.5:
                                inf, rec = si, "sillon"; break
                            if (ampl.var["zB"][p, d, si].value() or 0) > 0.5:
                                inf, rec = si, "cama"; break
                    vdur = next(t[2] for t in TOY_PATIENTS if t[0] == p)
                    espera = (inf - (vis + vdur)) if inf is not None else None
                    print(f"    P{p}: dia {d}, visita slot {vis}, infusion slot {inf}, "
                          f"{rec}, espera {espera}")
                    break


def main():
    modules.activate(AMPL_UUID)

    print("=" * 64)
    print("TOY PROBLEM — verificacion del modelo MILP")
    print("=" * 64)
    print(f"NV={NV} |SV|={len(SLOTS_VISITA)}   NI={NI} |SI|={len(SLOTS_INFUSION)}")
    print(f"Dias={DIAS}  Patologias={K}  Salas: {NSALAS}  Sillones={RSILL} Camas={RCAMA}")
    print(f"Pacientes: {len(TOY_PATIENTS)}")

    print("\nPREDICCION ANALITICA (calculada a mano):")
    print("  VISITA: pat1 tiene 1 sala -> 4 slots regulares (NV=4). 5 pacientes de pat1")
    print("    de 1 slot: 4 caben en slots 1..4, el 5º va al slot 5 (overtime de VISITA).")
    print("    Restriccion (9a): aV >= (5+1-1) - NV(4) = 1 slot.")
    print("  INFUSION: ese 5º paciente, al visitarse en slot 5, no puede infundirse antes")
    print("    del slot 6 (restriccion (5)/(6): la infusion empieza tras acabar la visita).")
    print("    Con f=2, su infusion ocupa los slots 6 y 7. Como NI=6, el slot 7 es overtime.")
    print("    Restriccion (9b/9c): aI >= (6+2-1) - NI(6) = 1 slot.")
    print("    >> El overtime de visita ARRASTRA overtime de infusion (acoplamiento")
    print("       de las dos etapas en un dia apretado).")
    print("  => F1 (overtime) = aV + aI = 1 + 1 = 2 slots.")
    print("  => F2 (espera)   = 0  (cada infusion arranca justo al acabar su visita).")
    print("  => F3 (sillon)   = 5  (los 5 no-criticos van a sillon; hay 2 sillones).")

    # ── P1: min overtime ────────────────────────────────────────────────────
    ampl, PB, PC = _build()
    ampl.eval("drop eps_f1; drop eps_f2;")
    ampl.eval("objective OvertimeTotal;")
    t0 = time.time(); ampl.solve(); t1 = time.time()
    f1 = _obj(ampl, "OvertimeTotal")
    print(f"\n[P1] {_status(ampl)} | F1 = {f1:.1f} slots overtime | {t1-t0:.2f}s")
    v1 = f1

    # ── P2: min espera, con F1 fijado ───────────────────────────────────────
    ampl, PB, PC = _build()
    ampl.eval("drop eps_f2;")
    ampl.param["v1_bar"] = v1
    ampl.eval("objective WaitTotal;")
    t0 = time.time(); ampl.solve(); t1 = time.time()
    f2 = _obj(ampl, "WaitTotal")
    print(f"[P2] {_status(ampl)} | F2 = {f2:.1f} slots espera | {t1-t0:.2f}s")
    v2 = f2

    # ── P3: max sillones, con F1 y F2 fijados ───────────────────────────────
    ampl, PB, PC = _build()
    ampl.param["v1_bar"] = v1
    ampl.param["v2_bar"] = v2
    ampl.eval("objective ChairsTotal;")
    t0 = time.time(); ampl.solve(); t1 = time.time()
    f3 = _obj(ampl, "ChairsTotal")
    print(f"[P3] {_status(ampl)} | F3 = {f3:.0f} pacientes en sillon | {t1-t0:.2f}s")

    _agenda(ampl, PB, PC)

    ok = (abs(f1 - 2) < 1e-6 and abs(f2 - 0) < 1e-6 and abs(f3 - 5) < 1e-6)
    print("\n" + "=" * 64)
    print("RESUMEN  (obtenido vs esperado)")
    print(f"  F1 = {f1:.1f}  (esperado 2: 1 visita + 1 infusion arrastrada)")
    print(f"  F2 = {f2:.1f}  (esperado 0)")
    print(f"  F3 = {f3:.0f}  (esperado 5)")
    print(f"  VERIFICACION: {'OK — el modelo da lo predicho' if ok else 'DISCREPANCIA — revisar'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
