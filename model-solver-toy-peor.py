# Toy "peor caso" para estresar el modelo y, sobre todo, el lower bound LB1.
#
# A diferencia de model-solver-toy.py (caso minimo, didactico, 1 dia, recursos con
# holgura), este toy reproduce a ESCALA REDUCIDA los tres ingredientes que hacen
# computacionalmente dificil el problema real:
#
#   1. Decision INTERDAY: varios dias, el modelo elige en que dia va cada paciente.
#   2. MCP RIGIDO con DEFICIT: una patologia con muchos pacientes y pocas salas, que
#      OBLIGA a usar overtime de visita (y, por arrastre, de infusion).
#   3. Recursos SATURADOS: pocos sillones/camas, de modo que la competencia por el
#      recurso puede forzar espera (F2 > 0) y reparto no trivial.
#
# Sigue siendo pequeno (resoluble en segundos), pero el valor de F1 se puede acotar
# analiticamente, que es lo que necesitamos para verificar LB1.
#
# El modelo AMPL es el mismo que model-solver-real.py (dimensiones reducidas).

import time

from amplpy import AMPL, modules

AMPL_UUID = ""

# ─────────────────────────────────────────────────────────────────────────────
# DIMENSIONES REDUCIDAS
# ─────────────────────────────────────────────────────────────────────────────
NV = 4
SLOTS_VISITA = list(range(1, 7))      # |SV| = 6  (4 regulares + 2 overtime)
NI = 6
SLOTS_INFUSION = list(range(1, 10))   # |SI| = 9  (6 regulares + 3 overtime)

DIAS = [1, 2, 3]      # INTERDAY: 3 dias -> el modelo decide el dia de cada paciente
K = [1, 2]            # dos patologias
RSILL = 2             # sillones (saturables)
RCAMA = 2             # camas

# MCP RIGIDO: 1 sala por patologia y dia (de las 6 fisicas, solo 2 en uso/dia).
# pat1 con 1 sala/dia y mucha demanda -> deficit -> overtime de visita forzado.
NSALAS = {(1, 1): 1, (1, 2): 1, (1, 3): 1,
          (2, 1): 1, (2, 2): 1, (2, 3): 1}

# ─────────────────────────────────────────────────────────────────────────────
# PACIENTES: pat1 saturada (fuerza overtime), pat2 holgada.
# ─────────────────────────────────────────────────────────────────────────────
# pat1: 15 pacientes v=1. Capacidad regular de visita = NV(4) x 3 dias = 12 slots.
#       15 > 12 -> al menos 3 pacientes deben ir a overtime de visita (slot 5+),
#       repartibles como 5+5+5 (1 por dia en slot 5). Cada uno arrastra ademas
#       overtime de infusion (inf en slots 6-7, NI=6). Cota: F1 >= 3 (solo visita).
# pat2: 6 pacientes v=1, 1 sala/dia: 2/dia caben en slots 1-2 sin overtime.
def _make_patients():
    pts = []
    pid = 1
    # pat1: 15 pacientes, mezcla critico/no-critico (f=2)
    for i in range(15):
        critico = 1 if i % 4 == 0 else 0   # ~1 de cada 4 critico
        pts.append((pid, 1, 1, 2, critico))
        pid += 1
    # pat2: 6 pacientes
    for i in range(6):
        critico = 1 if i % 3 == 0 else 0
        pts.append((pid, 2, 1, 2, critico))
        pid += 1
    return pts


TOY_PATIENTS = _make_patients()

# ─────────────────────────────────────────────────────────────────────────────
# MODELO AMPL (mismo que model-solver-real.py, dimensiones reducidas)
# ─────────────────────────────────────────────────────────────────────────────
AMPL_MODEL = """
set P; set PB; set PC; set D; set SV; set SI; set K;
param vp{P} integer >= 1;
param fp{P} integer >= 1;
param alpha{P} integer >= 1;
param nsalas{K, D} integer >= 0;
param NV integer; param NI integer; param RSILL integer; param RCAMA integer;
param v1_bar default 1e9; param v2_bar default 1e9;

var x{P, D, SV}  binary;
var y{PB, D, SI} binary;
var zB{PC, D, SI} binary;
var zC{PC, D, SI} binary;
var aV{D} >= 0; var aI{D} >= 0; var om{D} >= 0;

minimize OvertimeTotal: sum{d in D} (aV[d] + aI[d]);
minimize WaitTotal:     sum{d in D} om[d];
maximize ChairsTotal:   sum{p in PC, d in D, s in SI} zC[p,d,s];

subject to unicidad{p in P}:
    sum{d in D, s in SV} x[p,d,s] = 1;
subject to cap_salas{k in K, d in D, s in SV}:
    sum{p in P, q in SV: alpha[p] = k and q >= max(1, s-vp[p]+1) and q <= s} x[p,d,q]
    <= nsalas[k,d];
subject to mismo_dia_pb{p in PB, d in D}:
    sum{s in SV} x[p,d,s] = sum{s in SI} y[p,d,s];
subject to mismo_dia_pc{p in PC, d in D}:
    sum{s in SV} x[p,d,s] = sum{s in SI} (zB[p,d,s] + zC[p,d,s]);
subject to seq_pb{p in PB, d in D}:
    sum{s in SI} s * y[p,d,s] >= sum{s in SV} (s + vp[p]) * x[p,d,s];
subject to seq_pc{p in PC, d in D}:
    sum{s in SI} s * (zB[p,d,s] + zC[p,d,s]) >= sum{s in SV} (s + vp[p]) * x[p,d,s];
subject to cap_sillones{d in D, s in SI}:
    sum{p in PC, q in SI: q >= max(1, s-fp[p]+1) and q <= s} zC[p,d,q] <= RSILL;
subject to cap_camas{d in D, s in SI}:
    sum{p in PB, q in SI: q >= max(1, s-fp[p]+1) and q <= s} y[p,d,q]
  + sum{p in PC, q in SI: q >= max(1, s-fp[p]+1) and q <= s} zB[p,d,q] <= RCAMA;
subject to ot_visitas{p in P, d in D}:
    aV[d] >= sum{s in SV} (s + vp[p] - 1) * x[p,d,s] - NV;
subject to ot_inf_pb{p in PB, d in D}:
    aI[d] >= sum{s in SI} (s + fp[p] - 1) * y[p,d,s] - NI;
subject to ot_inf_pc{p in PC, d in D}:
    aI[d] >= sum{s in SI} (s + fp[p] - 1) * (zB[p,d,s] + zC[p,d,s]) - NI;
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
    ampl.param["vp"]    = {p[0]: p[2] for p in TOY_PATIENTS}
    ampl.param["fp"]    = {p[0]: p[3] for p in TOY_PATIENTS}
    ampl.param["alpha"] = {p[0]: p[1] for p in TOY_PATIENTS}
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


def compute_lb1():
    """Lower bound LB1 (Procedimiento 1 del articulo) adaptado al toy.

    Combina la cota de CONTEO (exceso de demanda de visita sobre capacidad regular,
    robusta) con la cota LP (relajacion continua, la que describe el articulo) y toma
    el maximo. En el toy la cota de conteo da 3.
    """
    P     = [p[0] for p in TOY_PATIENTS]
    alpha = {p[0]: p[1] for p in TOY_PATIENTS}
    v_p   = {p[0]: p[2] for p in TOY_PATIENTS}

    salas_k = {k: sum(NSALAS.get((k, d), 0) for d in DIAS) for k in K}
    dem_k   = {k: 0 for k in K}
    for p in P:
        dem_k[alpha[p]] += v_p[p]
    K_bar = [k for k in K if NV * salas_k[k] < dem_k[k]]
    if not K_bar:
        return 0.0, []

    # (a) cota de conteo
    lb_conteo = sum(max(0, dem_k[k] - NV * salas_k[k]) for k in K_bar)

    # (b) cota LP: relajacion continua sobre los pacientes en deficit
    PB = [p[0] for p in TOY_PATIENTS if p[4] == 1]
    PC = [p[0] for p in TOY_PATIENTS if p[4] == 0]
    P_bar  = [p for p in P if alpha[p] in K_bar]
    PB_bar = [p for p in P_bar if p in PB]
    PC_bar = [p for p in P_bar if p in PC]
    ampl = AMPL()
    ampl.eval(AMPL_MODEL)
    ampl.set["P"]  = P_bar
    ampl.set["PB"] = PB_bar
    ampl.set["PC"] = PC_bar
    ampl.set["D"]  = DIAS
    ampl.set["SV"] = SLOTS_VISITA
    ampl.set["SI"] = SLOTS_INFUSION
    ampl.set["K"]  = K
    ampl.param["NV"]    = NV
    ampl.param["NI"]    = NI
    ampl.param["RSILL"] = 10 ** 6
    ampl.param["RCAMA"] = 10 ** 6
    ampl.param["vp"]    = {p: v_p[p] for p in P_bar}
    ampl.param["fp"]    = {p[0]: p[3] for p in TOY_PATIENTS if p[0] in P_bar}
    ampl.param["alpha"] = {p: alpha[p] for p in P_bar}
    ampl.param["nsalas"] = {(k, d): NSALAS.get((k, d), 0) for k in K for d in DIAS}
    ampl.eval("drop eps_f1; drop eps_f2;")
    ampl.option["relax_integrality"] = 1
    ampl.eval("objective OvertimeTotal;")
    ampl.option["solver"] = "gurobi"
    ampl.option["gurobi_options"] = "outlev=0"
    ampl.solve()
    lb_lp = max(0.0, _obj(ampl, "OvertimeTotal") or 0.0)

    print(f"      (cota conteo={lb_conteo}  cota LP={lb_lp:.2f})")
    return max(lb_conteo, lb_lp), K_bar


def main():
    modules.activate(AMPL_UUID)

    npat = len(TOY_PATIENTS)
    ncrit = sum(1 for p in TOY_PATIENTS if p[4] == 1)
    print("=" * 64)
    print("TOY PEOR-CASO — estres del modelo (interday + MCP rigido + saturacion)")
    print("=" * 64)
    print(f"NV={NV} |SV|={len(SLOTS_VISITA)}   NI={NI} |SI|={len(SLOTS_INFUSION)}")
    print(f"Dias={DIAS}  Patologias={K}  Sillones={RSILL} Camas={RCAMA}")
    print(f"MCP nsalas: pat1=1 sala/dia, pat2=1 sala/dia")
    print(f"Pacientes: {npat}  ({ncrit} criticos)")

    print("\nPREDICCION ANALITICA del overtime (cota para verificar LB1):")
    print("  pat1: 15 pacientes v=1, 1 sala/dia. Capacidad visita regular = NV(4) x 3 dias = 12.")
    print("    15 > 12 -> al menos 3 pacientes en overtime de visita (reparto 5+5+5).")
    print("    -> overtime de VISITA >= 3 slots (1 por dia).")
    print("    Ademas cada paciente de slot 5 arrastra overtime de infusion (inf slots 6-7).")
    print("  pat2: 6 pacientes, 2/dia, caben sin overtime.")
    print("  => COTA INFERIOR esperada de F1 (solo visita) = 3.")
    print("     F1 real sera >= 3 (puede subir por el arrastre de infusion).")
    print("     [Esta es justo la cota que LB1 debe reproducir: LB1 = 3.]")

    # ── Verificacion de LB1 ──
    lb1, K_bar = compute_lb1()
    print(f"\n[LB1] calculado = {lb1:.1f}  (patologias en deficit: {K_bar})")
    print(f"[LB1] esperado  = 3.0  -> {'OK' if abs(lb1 - 3) < 1e-6 else 'REVISAR'}")

    # ── P1 (con y sin LB1, para medir el efecto en nodos de B&B) ──
    # outlev=1 para ver los "branching nodes" de Gurobi (efecto de LB1)
    print("\n[P1] sin LB1:")
    ampl, PB, PC = _build()
    ampl.eval("drop eps_f1; drop eps_f2;")
    ampl.eval("objective OvertimeTotal;")
    ampl.option["gurobi_options"] = "outlev=1"
    t0 = time.time(); ampl.solve(); t1 = time.time()
    f1 = _obj(ampl, "OvertimeTotal")
    print(f"[P1] {_status(ampl)} | F1 = {f1:.1f} slots overtime | {t1-t0:.2f}s")
    av = {d: max(0.0, ampl.var['aV'][d].value()) for d in DIAS}
    ai = {d: max(0.0, ampl.var['aI'][d].value()) for d in DIAS}
    print(f"     desglose por dia: aV={av}  aI={ai}")
    v1 = f1

    # ── P1 con LB1 (mismo modelo + desigualdad F1 >= LB1) ──
    print("\n[P1] con LB1 (desigualdad F1 >= LB1):")
    ampl, PB, PC = _build()
    ampl.eval("drop eps_f1; drop eps_f2;")
    ampl.eval("objective OvertimeTotal;")
    if lb1 > 0:
        ampl.eval(f"subject to lb1_cut: sum{{d in D}} (aV[d] + aI[d]) >= {lb1};")
    ampl.option["gurobi_options"] = "outlev=1"
    t0 = time.time(); ampl.solve(); t1 = time.time()
    f1b = _obj(ampl, "OvertimeTotal")
    print(f"[P1] {_status(ampl)} | F1 = {f1b:.1f} slots overtime | {t1-t0:.2f}s")
    print(f"     (compara los 'branching nodes' de Gurobi arriba: con LB1 deben bajar)")

    # ── P2 ──
    ampl, PB, PC = _build()
    ampl.eval("drop eps_f2;")
    ampl.param["v1_bar"] = v1
    ampl.eval("objective WaitTotal;")
    t0 = time.time(); ampl.solve(); t1 = time.time()
    f2 = _obj(ampl, "WaitTotal")
    print(f"[P2] {_status(ampl)} | F2 = {f2:.1f} slots espera | {t1-t0:.2f}s")
    v2 = f2

    # ── P3 ──
    ampl, PB, PC = _build()
    ampl.param["v1_bar"] = v1
    ampl.param["v2_bar"] = v2
    ampl.eval("objective ChairsTotal;")
    t0 = time.time(); ampl.solve(); t1 = time.time()
    f3 = _obj(ampl, "ChairsTotal")
    print(f"[P3] {_status(ampl)} | F3 = {f3:.0f} pacientes en sillon | {t1-t0:.2f}s")

    print("\n" + "=" * 64)
    print("RESUMEN")
    print(f"  F1 = {f1:.1f}  (cota inferior esperada por visitas = 3; F1 real >= 3)")
    print(f"  F2 = {f2:.1f}  (espera; >0 indica saturacion de recursos)")
    print(f"  F3 = {f3:.0f}  no-criticos en sillon")
    print(f"  VERIFICACION COTA: {'OK (F1 >= 3)' if f1 >= 3 - 1e-6 else 'REVISAR (F1 < 3)'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
