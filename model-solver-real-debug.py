# Variante de depuracion de model-solver-real.py
# - Resuelve UNICAMENTE las instancias indicadas (por defecto 11, 16, 17)
# - Redirige TODA la salida (incluida la traza completa de Gurobi en cada
#   sub-problema, tambien la descomposicion diaria de P2) al log, no solo el
#   resumen escueto.
# - Util para estudiar el comportamiento del modelo instancia a instancia y
#   para verificar el fix del mapeo c[i,j].
#
# Reutiliza el modelo y las rutinas de model-solver-real.py; solo cambia el
# bucle principal y fuerza outlev=1 en todos los sub-problemas.

import importlib.util
import os

import pandas as pd
from amplpy import modules

# El fichero principal se llama con guiones, no es importable directamente.
# Lo cargamos por ruta para reutilizar TODO su codigo sin duplicar el modelo AMPL.
_spec = importlib.util.spec_from_file_location("msr", "model-solver-real.py")
msr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(msr)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION DE DEPURACION
# ─────────────────────────────────────────────────────────────────────────────
# Filas 11, 16, 17 de la Tabla 7 del articulo. La numeracion .dat NO coincide
# con la del articulo; la correspondencia se fija por nº de pacientes:
#   Tabla 7 fila 11 (714 pac) = istanza11.dat
#   Tabla 7 fila 16 (588 pac) = istanza17.dat
#   Tabla 7 fila 17 (603 pac) = istanza18.dat
INSTANCIAS = ["istanza11", "istanza17", "istanza18"]  # modificar segun se quiera
LOGS_DIR   = "logs-real-debug"
OUT_DIR    = os.path.join("Databases", "optimized-sets-real-debug")


# ─────────────────────────────────────────────────────────────────────────────
# Forzar outlev=1 en la descomposicion diaria de P2 (en el solver normal va a 0)
# ─────────────────────────────────────────────────────────────────────────────
def _solve_day_p2_verbose(d, PB_d, PC_d, v_p, f_p, alpha_p, w_mcp, aV_fixed, aI_fixed):
    P_d = PB_d + PC_d
    if not P_d:
        return {"om": 0.0, "x": {}, "y": {}, "zB": {}, "zC": {}}

    ampl = msr._build_ampl([d], PB_d, PC_d, v_p, f_p, alpha_p, w_mcp)
    ampl.var["aV"][d].fix(aV_fixed)
    ampl.var["aI"][d].fix(aI_fixed)

    ampl.eval("drop eps_f1; drop eps_f2;")
    ampl.eval("objective WaitTotal;")
    # DIFERENCIA con el solver normal: outlev=1 para volcar la traza completa
    ampl.option["gurobi_options"] = (
        f"mipgap={msr.GAP_REL} outlev=1 timelim={msr.TIME_LIMIT_DAY}"
    )

    print(f"\n  >>> Sub-problema dia {d} ({len(P_d)} pacientes) <<<")
    ampl.solve()

    status = msr._solve_status(ampl)
    if "infeasible" in status.lower():
        print(f"  [WARN] sub-problema dia {d} infactible — warm start parcial")
        return {"om": 0.0, "x": {}, "y": {}, "zB": {}, "zC": {}}

    om_val = max(0.0, ampl.var["om"][d].value())
    return {
        "om":  om_val,
        "x":   msr._read_x(ampl),
        "y":   msr._read_var3(ampl, "y",  PB_d),
        "zB":  msr._read_var3(ampl, "zB", PC_d),
        "zC":  msr._read_var3(ampl, "zC", PC_d),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BUCLE PRINCIPAL — solo las instancias indicadas, log completo
# ─────────────────────────────────────────────────────────────────────────────
def run_debug():
    modules.activate(msr.AMPL_UUID)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Parchear la descomposicion diaria para que sea verbosa
    msr._solve_day_p2 = _solve_day_p2_verbose

    # Opcion C: MCP mensual fijo. La agrupacion en meses necesita TODAS las instancias,
    # no solo las 3 de depuracion -> se construye sobre el directorio completo.
    dat_files  = sorted(msr.glob(os.path.join(msr.REAL_DIR, "*.dat")), key=msr._natural_key)
    period_mcp = msr.build_period_mcps(dat_files)

    resumen = []

    for stem in INSTANCIAS:
        dat_path = os.path.join(msr.REAL_DIR, f"{stem}.dat")
        if not os.path.exists(dat_path):
            print(f"[SKIP] no existe {dat_path}")
            continue

        log_path = os.path.join(LOGS_DIR, f"run_{stem}_debug.txt")

        with msr.Tee(log_path):
            print("=" * 70)
            print(f"INSTANCIA : {stem}  [DEBUG — datos reales, AMPL + Gurobi]")
            print(f"FICHERO   : {dat_path}")
            print(f"FECHA     : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)

            dat = msr.parse_dat(dat_path)
            dias, PB, PC, v_p, f_p, alpha_p, nsalas = msr.prepare_instance(dat, period_mcp[stem])

            print(f"giornilav = {dat['giornilav']}  |  ptot = {len(dat['patients'])}")
            print(f"c_map (dia laborable -> dia natural): {dat['c_map'] or 'identidad'}")
            print("nsalas[k,d] (salas por patologia y dia laborable, festivos acumulados):")
            for k in range(1, 8):
                fila = {d: nsalas.get((k, d), 0) for d in dias}
                print(f"   pat{k}: {fila}")

            res = msr.build_and_solve(dias, PB, PC, v_p, f_p, alpha_p, nsalas)

            out_path = None
            if res.get("feasible"):
                stem_out = os.path.join(OUT_DIR, f"agenda_{stem}.csv")
                res["results_df"].sort_values(
                    ["Dia", "Slot_Inicio_Visita"]
                ).to_csv(stem_out, index=False)
                out_path = stem_out
                print(f"Agenda guardada: {out_path}")
            else:
                print("[WARN] Instancia sin solucion coherente — no se guarda agenda.")

        resumen.append({
            "instancia":         stem,
            "giornilav":         dat["giornilav"],
            "n_pacientes":       res.get("n_patients", 0),
            "n_criticos":        res.get("n_critical", 0),
            "status_p1":         res.get("status_p1",  "N/A"),
            "status_p2":         res.get("status_p2",  "N/A"),
            "status_p3":         res.get("status_p3",  "N/A"),
            "f1_overtime_slots": res.get("f1",         float("nan")),
            "f2_maxwait_slots":  res.get("f2",         float("nan")),
            "f3_sillones":       res.get("f3",         float("nan")),
            "pct_sillon":        res.get("pct_sillon", float("nan")),
            "espera_media_min":  res.get("mean_wait",  float("nan")),
            "t_total_s":         res.get("elapsed",    float("nan")),
            "coherente":         res.get("coherent",   False),
            "log":               log_path,
            "agenda":            out_path or "",
        })

    df_res   = pd.DataFrame(resumen)
    res_path = os.path.join(OUT_DIR, "resumen_debug.csv")
    df_res.to_csv(res_path, index=False)
    print(f"\nResumen debug guardado en: {res_path}")


if __name__ == "__main__":
    run_debug()
