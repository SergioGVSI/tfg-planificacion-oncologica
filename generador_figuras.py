# -*- coding: utf-8 -*-
"""
Generador de figuras para la memoria del TFG
============================================
Clase `GeneradorFiguras`: produce TODAS las ilustraciones (gráficas, mapas de
calor, esquemas conceptuales y mapas de ideas) a partir de los datos del
proyecto, para insertarlas en la memoria.

Fuentes de datos (rutas relativas a la raíz del repositorio):
  - Databases/real-sets/*.dat .............. instancias reales crudas (entrada)
  - Databases/optimized-sets-real/ ......... resultados Opción C (resumen + agendas)
  - logs-real/run_*.txt .................... logs de Gurobi (Opción C)
  - Databases/optimized-sets-real-crudo/ ... resultados MCP crudo (resumen + agendas)
  - logs-real-crudo/run_*.txt ............. logs de Gurobi (MCP crudo)
  - Databases/optimized-sets/ .............. resultados sobre datos SIMULADOS (90 runs)

Datos de referencia del artículo (Carello, Passacantando & Tánfani, EJOR 2025)
embebidos: Anexo I / Tabla 12 del borrador (= Tabla 7 del artículo) y Tabla 8
(utilización de salas por patología).

Uso:
    python generador_figuras.py            # genera todo en ./figuras/
    # o desde código:
    g = GeneradorFiguras(); g.generar_todo()
    g.fig_A1_overtime_real()               # una figura concreta

Dependencias: matplotlib, pandas, numpy.
"""

import os
import re
from glob import glob
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # sin pantalla
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DEL PROBLEMA
# ─────────────────────────────────────────────────────────────────────────────
NV = 36           # último slot regular de visita (14:00)
NI = 54           # último slot regular de infusión (17:00)
SV_MAX = 48       # |S^V|
SI_MAX = 66       # |S^I|
MAX_SILLONES = 26
MAX_CAMAS = 27
PATOLOGIAS = {1: "HE", 2: "GI", 3: "UR", 4: "GY", 5: "BR", 6: "OT", 7: "LU"}
NOMBRES_PAT = {1: "Hematología", 2: "Gastrointestinal", 3: "Urología",
               4: "Ginecología", 5: "Mama", 6: "Otros", 7: "Pulmón"}

# Paleta apta para daltónicos (Okabe-Ito)
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
         "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
COL_C = "#0072B2"        # Opción C / nuestro
COL_ART = "#E69F00"      # artículo
COL_CRUDO = "#D55E00"    # MCP crudo
COL_OK = "#009E73"
COL_BAD = "#D55E00"


# ─────────────────────────────────────────────────────────────────────────────
# DATOS DE REFERENCIA DEL ARTÍCULO (Anexo I / Tabla 12 del borrador = Tabla 7)
# fila: (pac_total, criticos, overtime_F1, pref_P3_%sillon)
# ─────────────────────────────────────────────────────────────────────────────
ART_TABLA12 = {
    1: (528, 128, 0, 85.50), 2: (665, 196, 0, 88.91), 3: (642, 176, 0, 89.48),
    4: (554, 135, 0, 97.37), 5: (635, 172, 0, 89.20), 6: (637, 182, 0, 93.85),
    7: (656, 175, 0, 87.73), 8: (587, 175, 0, 99.27), 9: (609, 156, 0, 93.16),
    10: (687, 182, 0, 84.75), 11: (714, 186, 8, 81.25), 12: (609, 168, 0, 95.46),
    13: (658, 168, 0, 86.12), 14: (699, 179, 0, 82.69), 15: (607, 140, 0, 92.51),
    16: (588, 149, 8, 82.23), 17: (603, 139, 6, 76.94), 18: (719, 179, 0, 80.00),
    19: (591, 147, 0, 97.30), 20: (623, 147, 0, 89.08), 21: (667, 169, 0, 86.35),
    22: (655, 169, 0, 86.63), 23: (621, 154, 0, 91.01), 24: (634, 149, 0, 86.19),
    25: (608, 158, 0, 92.89), 26: (647, 164, 0, 87.99), 27: (649, 140, 0, 83.69),
    28: (606, 148, 0, 91.92), 29: (595, 138, 0, 91.68), 30: (642, 141, 0, 85.63),
    31: (585, 142, 0, 97.29), 32: (431, 81, 0, 97.43), 33: (653, 153, 0, 84.80),
    34: (608, 141, 0, 91.65), 35: (573, 128, 0, 92.36), 36: (560, 108, 0, 91.15),
    37: (583, 139, 0, 91.67), 38: (623, 157, 0, 88.84), 39: (626, 162, 0, 91.59),
    40: (602, 121, 0, 89.60), 41: (612, 150, 0, 94.37), 42: (633, 163, 0, 90.64),
    43: (539, 126, 0, 83.78), 44: (649, 148, 0, 82.24), 45: (642, 162, 0, 85.00),
    46: (557, 125, 0, 92.82), 47: (577, 134, 0, 92.55), 48: (670, 148, 0, 82.95),
    49: (617, 136, 0, 88.57), 50: (625, 138, 0, 87.06), 51: (430, 92, 0, 76.63),
}
ART_PROMEDIO_SILLON = 88.82
ART_PROMEDIO_OVERTIME = 0.43

# Tabla 8 del artículo: utilización de salas por patología (media / mín / máx %)
ART_TABLA8 = {
    "Global": (68.52, 56.96, 85.89), "HE": (86.89, 62.50, 99.50),
    "GI": (72.44, 45.40, 100.00), "UR": (54.20, 36.10, 108.30),
    "GY": (34.80, 10.60, 102.80), "BR": (83.37, 63.90, 106.90),
    "OT": (74.28, 59.30, 115.30), "LU": (73.69, 43.10, 97.20),
}


def _stem_num(stem):
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else 10 ** 9


def art_row_for(stem):
    """Mapea istanzaNN -> fila de la Tabla 12 del artículo.
    istanza01-13 -> filas 1-13; istanza15-52 -> filas 14-51; istanza14 excluida."""
    n = _stem_num(stem)
    if n == 14:
        return None
    return n if n <= 13 else n - 1


# ─────────────────────────────────────────────────────────────────────────────
class GeneradorFiguras:
    def __init__(self, root=".", outdir="figuras", formatos=("pdf", "png")):
        self.root = root
        self.outdir = os.path.join(root, outdir)
        self.formatos = formatos
        os.makedirs(self.outdir, exist_ok=True)

        self.DIR_REAL = os.path.join(root, "Databases", "real-sets")
        self.DIR_OPT_C = os.path.join(root, "Databases", "optimized-sets-real")
        self.DIR_OPT_CRUDO = os.path.join(root, "Databases", "optimized-sets-real-crudo")
        self.DIR_OPT_SIM = os.path.join(root, "Databases", "optimized-sets")
        self.DIR_LOGS_C = os.path.join(root, "logs-real")
        self.DIR_LOGS_CRUDO = os.path.join(root, "logs-real-crudo")

        self._cache = {}
        self._estilo()

    # ───────────────────────────────── estilo ──────────────────────────────
    def _estilo(self):
        plt.rcParams.update({
            "figure.figsize": (7.2, 4.3),
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.axisbelow": True,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
        })

    def _save(self, fig, name):
        for ext in self.formatos:
            fig.savefig(os.path.join(self.outdir, f"{name}.{ext}"))
        plt.close(fig)
        print(f"  [OK] {name}")

    @staticmethod
    def _slot2hora(slot, base_h=8):
        minutos = int((slot - 1) * 10)
        h = base_h + minutos // 60
        m = minutos % 60
        return f"{h:02d}:{m:02d}"

    # ───────────────────────────────── loaders ─────────────────────────────
    def _resumen_real(self):
        if "real" not in self._cache:
            df = pd.read_csv(os.path.join(self.DIR_OPT_C, "resumen_ejecucion_real.csv"))
            df = df.sort_values("instancia", key=lambda s: s.map(_stem_num)).reset_index(drop=True)
            self._cache["real"] = df
        return self._cache["real"]

    def _resumen_crudo(self):
        if "crudo" not in self._cache:
            p = os.path.join(self.DIR_OPT_CRUDO, "resumen_ejecucion_real_crudo.csv")
            df = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
            if not df.empty:
                df = df.sort_values("instancia", key=lambda s: s.map(_stem_num)).reset_index(drop=True)
            self._cache["crudo"] = df
        return self._cache["crudo"]

    def _resumen_sim(self):
        if "sim" not in self._cache:
            df = pd.read_csv(os.path.join(self.DIR_OPT_SIM, "resumen_ejecucion_ampl.csv"))
            self._cache["sim"] = df
        return self._cache["sim"]

    def _parse_dat(self, path):
        with open(path, "r", encoding="utf-8") as fh:
            t = fh.read()
        g = int(re.search(r"param\s+giornilav\s*:=\s*(\d+)", t).group(1))
        c_map = {}
        cb = re.search(r"param\s+c\s*:\s*([\d\s]+?):=(.*?);", t, re.DOTALL)
        if cb:
            cols = [int(x) for x in cb.group(1).split()]
            for row in cb.group(2).strip().splitlines():
                v = row.split()
                if not v or not v[0].isdigit():
                    continue
                i = int(v[0])
                for c, f in zip(cols, v[1:]):
                    if f == "1":
                        c_map[i] = c
                        break
        w = {}
        wb = re.search(r"param\s+w\s*:=(.*?);", t, re.DOTALL)
        for e in re.finditer(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s+1", wb.group(1)):
            w[(int(e.group(1)), int(e.group(2)), int(e.group(3)))] = 1
        pats = {}
        pb = re.search(r"param\s*:\s*alpha\s+v\s+f\s+lambda\s*:=(.*?);", t, re.DOTALL)
        for row in re.finditer(r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", pb.group(1)):
            pats[int(row.group(1))] = dict(alpha=int(row.group(2)), v=int(row.group(3)),
                                           f=int(row.group(4)), lam=int(row.group(5)))
        return dict(giornilav=g, c_map=c_map, w=w, patients=pats)

    def _all_dats(self):
        """Lista de (stem, dat) de las 52 instancias, ordenadas, excluyendo .DS_Store."""
        if "dats" not in self._cache:
            paths = sorted(glob(os.path.join(self.DIR_REAL, "istanza*.dat")), key=_stem_num)
            self._cache["dats"] = [(os.path.splitext(os.path.basename(p))[0], self._parse_dat(p))
                                   for p in paths]
        return self._cache["dats"]

    def _agenda(self, stem, crudo=False):
        d = self.DIR_OPT_CRUDO if crudo else self.DIR_OPT_C
        p = os.path.join(d, f"agenda_{stem}.csv")
        return pd.read_csv(p) if os.path.exists(p) else None

    def _agenda_vf(self, stem, crudo=False):
        """Agenda enriquecida con v y f (de cada paciente, leídos del .dat)."""
        ag = self._agenda(stem, crudo=crudo)
        if ag is None or ag.empty:
            return None
        dat = self._parse_dat(os.path.join(self.DIR_REAL, f"{stem}.dat"))
        pats = dat["patients"]
        ag = ag.copy()
        ag["v"] = ag["ID_Paciente"].map(lambda i: pats.get(int(i), {}).get("v", np.nan))
        ag["f"] = ag["ID_Paciente"].map(lambda i: pats.get(int(i), {}).get("f", np.nan))
        ag["alpha"] = ag["ID_Paciente"].map(lambda i: pats.get(int(i), {}).get("alpha", np.nan))
        return ag

    def _parse_log(self, stem, crudo=False):
        """Extrae 'interioridades' del log de Gurobi: LB1, conteo, nodos B&B de P1,
        estado de P1 y gap final de P3. Devuelve dict con NaN donde no aplique."""
        d = self.DIR_LOGS_CRUDO if crudo else self.DIR_LOGS_C
        p = os.path.join(d, f"run_{stem}.txt")
        out = dict(lb1=np.nan, conteo=np.nan, p1_nodes=np.nan,
                   p1_status=None, p3_gap=np.nan, f1=np.nan)
        if not os.path.exists(p):
            return out
        t = open(p, "r", encoding="utf-8", errors="ignore").read()
        m = re.search(r"\[P1\]\s+LB1\s*=\s*([\d.]+)\s+\(LP reforzado\)\s+\[conteo_xcheck=([\d.]+)\]", t)
        if m:
            out["lb1"] = float(m.group(1)); out["conteo"] = float(m.group(2))
        # trocear por subproblema
        idx_p2 = t.find("[P2] min espera")
        idx_p3 = t.find("[P3] max sillones")
        chunk_p1 = t[:idx_p2] if idx_p2 > 0 else t
        chunk_p3 = t[idx_p3:] if idx_p3 > 0 else ""
        mn = re.findall(r"(\d+)\s+branching nodes", chunk_p1)
        if mn:
            out["p1_nodes"] = int(mn[-1])
        ms = re.search(r"\[P1\]\s+(solved|limit)\s+\|\s+F1\s*=\s*([\d.]+)", chunk_p1)
        if ms:
            out["p1_status"] = ms.group(1); out["f1"] = float(ms.group(2))
        mg = re.search(r"gap\s+([\d.]+)%", chunk_p3)
        if mg:
            out["p3_gap"] = float(mg.group(1))
        return out

    # ─────────────── Opción C: dotación de salas por patología/periodo ──────
    @staticmethod
    def _alloc_salas(demanda_k):
        salas = {k: 0 for k in range(1, 8)}
        defr = lambda k: demanda_k.get(k, 0) - NV * salas[k]
        for _ in range(30):
            cd = [k for k in range(1, 8) if defr(k) > 0]
            kb = max(cd, key=defr) if cd else max(range(1, 8), key=lambda kk: demanda_k.get(kk, 0))
            salas[kb] += 1
        return salas

    def _opcionC_salas_por_stem(self):
        """Replica el dimensionado mensual de la Opción C: agrupa por firma de MCP
        (=mes), dimensiona al pico de demanda del periodo. Devuelve {stem: salas_k}."""
        if "salasC" in self._cache:
            return self._cache["salasC"]
        info = []
        for stem, dat in self._all_dats():
            dem = Counter()
            for p in dat["patients"].values():
                dem[p["alpha"]] += p["v"]
            firma = Counter()
            for (_r, k, _d) in dat["w"]:
                firma[k] += 1
            info.append((stem, tuple(firma[k] for k in range(1, 8)), dem))
        # periodos por firma consecutiva
        periodos = []
        for stem, fm, dem in info:
            if not periodos or periodos[-1][0] != fm:
                periodos.append((fm, []))
            periodos[-1][1].append((stem, dem))
        salasC = {}
        for fm, weeks in periodos:
            pico = {k: max(dem[k] for _, dem in weeks) for k in range(1, 8)}
            al = self._alloc_salas(pico)
            for stem, _dem in weeks:
                salasC[stem] = al
        self._cache["salasC"] = salasC
        return salasC

    # ════════════════════════════════════════════════════════════════════════
    # GRUPO A — RESULTADOS REALES (resumen Opción C, 51 instancias)
    # ════════════════════════════════════════════════════════════════════════
    def fig_A1_overtime_real(self):
        df = self._resumen_real()
        fig, ax = plt.subplots(figsize=(9, 3.6))
        x = np.arange(len(df))
        f1 = df["f1_overtime_slots"].values
        colores = [COL_BAD if v > 0 else COL_C for v in f1]
        ax.bar(x, f1, color=colores)
        for i, v in enumerate(f1):
            if v > 0:
                ax.annotate(f"{v:.0f}", (i, v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("istanza", "") for s in df["instancia"]],
                           rotation=90, fontsize=6)
        ax.set_xlabel("Instancia real")
        ax.set_ylabel("Overtime $F_1$ (slots)")
        ax.set_title("Overtime por instancia (Opción C): 0 en 49/51")
        ax.set_ylim(0, max(4, f1.max() * 1.25))
        leg = [Line2D([0], [0], color=COL_C, lw=6, label="$F_1=0$"),
               Line2D([0], [0], color=COL_BAD, lw=6, label="$F_1>0$ (festivas 17/18)")]
        ax.legend(handles=leg, loc="upper right")
        self._save(fig, "A1_overtime_real")

    def fig_A2_sillon_vs_articulo(self):
        df = self._resumen_real()
        rows = []
        for _, r in df.iterrows():
            row = art_row_for(r["instancia"])
            if row is None:
                continue
            rows.append((r["instancia"], r["pct_sillon"], ART_TABLA12[row][3]))
        d = pd.DataFrame(rows, columns=["inst", "nuestro", "art"])
        x = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.plot(x, d["art"], "-o", color=COL_ART, ms=3, lw=1, label="Artículo (Tabla 7)")
        ax.plot(x, d["nuestro"], "-o", color=COL_C, ms=3, lw=1, label="Opción C (nuestro)")
        ax.axhline(d["nuestro"].mean(), color=COL_C, ls=":", lw=1, alpha=0.7)
        ax.axhline(ART_PROMEDIO_SILLON, color=COL_ART, ls=":", lw=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("istanza", "") for s in d["inst"]], rotation=90, fontsize=6)
        ax.set_xlabel("Instancia real")
        ax.set_ylabel("% pacientes en sillón ($F_3$)")
        ax.set_title(f"% sillón: nuestro (media {d['nuestro'].mean():.1f}%) vs artículo ({ART_PROMEDIO_SILLON}%)")
        ax.legend(loc="lower left")
        self._save(fig, "A2_sillon_vs_articulo")

    def fig_A3_delta_sillon(self):
        df = self._resumen_real()
        rows = []
        for _, r in df.iterrows():
            row = art_row_for(r["instancia"])
            if row is None:
                continue
            rows.append((r["instancia"], r["pct_sillon"] - ART_TABLA12[row][3]))
        d = pd.DataFrame(rows, columns=["inst", "delta"])
        x = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(9, 3.6))
        col = [COL_OK if v >= 0 else COL_BAD for v in d["delta"]]
        ax.bar(x, d["delta"], color=col)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(d["delta"].mean(), color="gray", ls="--", lw=1,
                   label=f"media {d['delta'].mean():.2f} pp")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("istanza", "") for s in d["inst"]], rotation=90, fontsize=6)
        ax.set_xlabel("Instancia real")
        ax.set_ylabel("Δ % sillón (nuestro − artículo)")
        ax.set_title("Desviación en % sillón frente al artículo (gap de $P_3$)")
        ax.legend()
        self._save(fig, "A3_delta_sillon")

    def fig_A4_tiempos_apilados(self):
        df = self._resumen_real()
        x = np.arange(len(df))
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.bar(x, df["t_p1_s"], color=OKABE[0], label="$P_1$")
        ax.bar(x, df["t_p2_s"], bottom=df["t_p1_s"], color=OKABE[1], label="$P_2$")
        ax.bar(x, df["t_p3_s"], bottom=df["t_p1_s"] + df["t_p2_s"], color=OKABE[2], label="$P_3$")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("istanza", "") for s in df["instancia"]], rotation=90, fontsize=6)
        ax.set_xlabel("Instancia real")
        ax.set_ylabel("Tiempo (s)")
        ax.set_title("Tiempo de resolución por subproblema (Opción C)")
        ax.legend(loc="upper left", ncol=3)
        self._save(fig, "A4_tiempos_apilados")

    def fig_A5_boxplot_tiempos(self):
        df = self._resumen_real()
        fig, ax = plt.subplots(figsize=(5.5, 4))
        data = [df["t_p1_s"], df["t_p2_s"], df["t_p3_s"]]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=["$P_1$", "$P_2$", "$P_3$"],
                        showmeans=True)
        for patch, c in zip(bp["boxes"], OKABE[:3]):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_ylabel("Tiempo (s)")
        ax.set_title("Distribución de tiempos por subproblema")
        ax.text(0.98, 0.95, "$P_3$ es el cuello de botella", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, style="italic")
        self._save(fig, "A5_boxplot_tiempos")

    def fig_A6_tiempo_vs_pacientes(self):
        df = self._resumen_real()
        fig, ax = plt.subplots(figsize=(6, 4))
        for gl, c in [(5, COL_C), (4, COL_ART), (3, COL_BAD)]:
            sub = df[df["giornilav"] == gl]
            if not sub.empty:
                ax.scatter(sub["n_pacientes"], sub["t_total_s"], color=c, s=28,
                           label=f"{gl} días", alpha=0.8, edgecolor="white", lw=0.5)
        ax.set_xlabel("Nº de pacientes en la semana")
        ax.set_ylabel("Tiempo total (s)")
        ax.set_title("Tiempo total vs tamaño de la instancia")
        ax.legend(title="Días laborables")
        self._save(fig, "A6_tiempo_vs_pacientes")

    def fig_A7_hist_sillon(self):
        df = self._resumen_real()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["pct_sillon"], bins=12, color=COL_C, alpha=0.8, edgecolor="white")
        ax.axvline(df["pct_sillon"].mean(), color=COL_C, ls="--",
                   label=f"media nuestro {df['pct_sillon'].mean():.1f}%")
        ax.axvline(ART_PROMEDIO_SILLON, color=COL_ART, ls="--",
                   label=f"media artículo {ART_PROMEDIO_SILLON}%")
        ax.set_xlabel("% pacientes en sillón ($F_3$)")
        ax.set_ylabel("Nº de instancias")
        ax.set_title("Distribución del % de sillón (Opción C)")
        ax.legend()
        self._save(fig, "A7_hist_sillon")

    def fig_A8_heatmap_status(self):
        df = self._resumen_real()
        mapa = {"solved": 1, "limit": 0}
        M = np.array([[mapa.get(df.iloc[i][f"status_p{j}"], 0) for j in (1, 2, 3)]
                      for i in range(len(df))])
        fig, ax = plt.subplots(figsize=(4.5, 9))
        ax.imshow(M, aspect="auto", cmap=matplotlib.colors.ListedColormap([COL_BAD, COL_OK]),
                  vmin=0, vmax=1)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["$P_1$", "$P_2$", "$P_3$"])
        ax.set_yticks(np.arange(len(df)))
        ax.set_yticklabels([s.replace("istanza", "") for s in df["instancia"]], fontsize=6)
        ax.set_title("Estado de cierre por subproblema\n(verde=óptimo ≤5%, rojo=límite)")
        ax.grid(False)
        self._save(fig, "A8_heatmap_status")

    def fig_A9_cierres(self):
        df = self._resumen_real()
        cierres = [(df[f"status_p{j}"] == "solved").sum() for j in (1, 2, 3)]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        b = ax.bar(["$P_1$", "$P_2$", "$P_3$"], cierres, color=OKABE[:3], alpha=0.85)
        for rect, v in zip(b, cierres):
            ax.annotate(f"{v}/{len(df)}", (rect.get_x() + rect.get_width() / 2, v),
                        ha="center", va="bottom")
        ax.set_ylabel("Instancias cerradas a óptimo (≤5%)")
        ax.set_title("Cierre por subproblema (el artículo cierra $P_3$ en 9/51)")
        ax.set_ylim(0, len(df) * 1.12)
        self._save(fig, "A9_cierres")

    def fig_A10_pacientes_criticos(self):
        df = self._resumen_real()
        x = np.arange(len(df))
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.bar(x, df["n_criticos"], color=COL_ART, label="Críticos (cama)")
        ax.bar(x, df["n_pacientes"] - df["n_criticos"], bottom=df["n_criticos"],
               color=COL_C, label="No críticos (sillón/cama)")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("istanza", "") for s in df["instancia"]], rotation=90, fontsize=6)
        ax.set_xlabel("Instancia real")
        ax.set_ylabel("Nº de pacientes")
        ax.set_title("Composición de pacientes por instancia")
        ax.legend(loc="upper right", ncol=2)
        self._save(fig, "A10_pacientes_criticos")

    # ════════════════════════════════════════════════════════════════════════
    # GRUPO B — AGENDAS (horarios por paciente)
    # ════════════════════════════════════════════════════════════════════════
    def _ocupacion(self, inicios, duraciones, smax):
        occ = np.zeros(smax + 2, dtype=int)
        for s0, dur in zip(inicios, duraciones):
            if np.isnan(s0) or np.isnan(dur):
                continue
            for s in range(int(s0), int(s0 + dur)):
                if 1 <= s <= smax:
                    occ[s] += 1
        return occ[1:smax + 1]

    def fig_B11_ocupacion_salas(self, stem="istanza03", dia=1):
        ag = self._agenda_vf(stem)
        if ag is None:
            return
        sub = ag[ag["Dia"] == dia]
        occ = self._ocupacion(sub["Slot_Inicio_Visita"], sub["v"], SV_MAX)
        x = np.arange(1, SV_MAX + 1)
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.bar(x, occ, color=COL_C, width=0.9)
        ax.axvline(NV + 0.5, color=COL_BAD, ls="--", lw=1.3, label="14:00 (fin regular)")
        ax.set_xlabel("Slot de visita (10 min)")
        ax.set_ylabel("Visitas simultáneas (salas en uso)")
        ax.set_title(f"Ocupación de salas a lo largo del día · {stem}, día {dia}")
        ax.set_xticks([1, 12, 24, 36, 48])
        ax.set_xticklabels([self._slot2hora(s) for s in [1, 12, 24, 36, 48]])
        ax.legend()
        self._save(fig, f"B11_ocupacion_salas_{stem}_d{dia}")

    def fig_B12_hist_inicios_visita(self, stem="istanza03"):
        ag = self._agenda_vf(stem)
        if ag is None:
            return
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.hist(ag["Slot_Inicio_Visita"], bins=np.arange(1, SV_MAX + 2), color=COL_C,
                alpha=0.85, edgecolor="white")
        ax.axvline(NV + 0.5, color=COL_BAD, ls="--", lw=1.3, label="14:00 (fin regular)")
        ax.set_xlabel("Slot de inicio de visita")
        ax.set_ylabel("Nº de pacientes")
        ax.set_title(f"Inicios de visita: ventana regular + cola de overtime · {stem}")
        ax.set_xticks([1, 12, 24, 36, 48])
        ax.set_xticklabels([self._slot2hora(s) for s in [1, 12, 24, 36, 48]])
        ax.legend()
        self._save(fig, f"B12_hist_inicios_visita_{stem}")

    def fig_B13_ocupacion_recursos(self, stem="istanza03", dia=1):
        ag = self._agenda_vf(stem)
        if ag is None:
            return
        sub = ag[ag["Dia"] == dia]
        sill = sub[sub["Recurso_Infusion"] == "sillon"]
        cama = sub[sub["Recurso_Infusion"] == "cama"]
        occ_s = self._ocupacion(sill["Slot_Inicio_Infusion"], sill["f"], SI_MAX)
        occ_c = self._ocupacion(cama["Slot_Inicio_Infusion"], cama["f"], SI_MAX)
        x = np.arange(1, SI_MAX + 1)
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(x, occ_s, color=COL_C, lw=1.6, label="Sillones en uso")
        ax.plot(x, occ_c, color=COL_ART, lw=1.6, label="Camas en uso")
        ax.axhline(MAX_SILLONES, color=COL_C, ls=":", lw=1, label=f"capacidad sillones ({MAX_SILLONES})")
        ax.axhline(MAX_CAMAS, color=COL_ART, ls=":", lw=1, label=f"capacidad camas ({MAX_CAMAS})")
        ax.axvline(NI + 0.5, color="gray", ls="--", lw=1, label="17:00 (fin regular)")
        ax.set_xlabel("Slot de infusión (10 min)")
        ax.set_ylabel("Recursos ocupados")
        ax.set_title(f"Ocupación de sillones y camas · {stem}, día {dia}")
        ax.set_xticks([1, 18, 36, 54, 66])
        ax.set_xticklabels([self._slot2hora(s) for s in [1, 18, 36, 54, 66]])
        ax.legend(fontsize=7, ncol=2)
        self._save(fig, f"B13_ocupacion_recursos_{stem}_d{dia}")

    def fig_B14_heatmap_dia_slot(self, stem="istanza03"):
        ag = self._agenda_vf(stem)
        if ag is None:
            return
        dias = sorted(ag["Dia"].unique())
        M = np.zeros((len(dias), SV_MAX))
        for i, d in enumerate(dias):
            sub = ag[ag["Dia"] == d]
            M[i] = self._ocupacion(sub["Slot_Inicio_Visita"], sub["v"], SV_MAX)
        fig, ax = plt.subplots(figsize=(9, 3))
        im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower",
                       extent=[0.5, SV_MAX + 0.5, dias[0] - 0.5, dias[-1] + 0.5])
        ax.axvline(NV + 0.5, color="white", ls="--", lw=1.2)
        ax.set_yticks(dias); ax.set_ylabel("Día laborable")
        ax.set_xlabel("Slot de visita")
        ax.set_xticks([1, 12, 24, 36, 48])
        ax.set_xticklabels([self._slot2hora(s) for s in [1, 12, 24, 36, 48]])
        ax.set_title(f"Mapa de carga de salas (día × slot) · {stem}")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="visitas simultáneas")
        self._save(fig, f"B14_heatmap_dia_slot_{stem}")

    def fig_B15_sillon_vs_cama(self):
        rows = []
        for stem, _ in self._all_dats():
            ag = self._agenda(stem)
            if ag is None:
                continue
            nc = ag[ag["Critico"] == 0]
            if nc.empty:
                continue
            sil = (nc["Recurso_Infusion"] == "sillon").sum()
            cam = (nc["Recurso_Infusion"] == "cama").sum()
            rows.append((stem, sil, cam))
        d = pd.DataFrame(rows, columns=["inst", "sil", "cam"])
        x = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.bar(x, d["sil"], color=COL_C, label="Sillón (preferido)")
        ax.bar(x, d["cam"], bottom=d["sil"], color=COL_ART, label="Cama (no críticos)")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("istanza", "") for s in d["inst"]], rotation=90, fontsize=6)
        ax.set_xlabel("Instancia real")
        ax.set_ylabel("Pacientes no críticos")
        ax.set_title("Reparto sillón vs cama en pacientes no críticos")
        ax.legend(ncol=2)
        self._save(fig, "B15_sillon_vs_cama")

    def fig_B18_pacientes_por_dia(self, stem="istanza03"):
        ag = self._agenda(stem)
        if ag is None:
            return
        cnt = ag.groupby("Dia").size()
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.bar(cnt.index, cnt.values, color=COL_C)
        for d, v in cnt.items():
            ax.annotate(str(v), (d, v), ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("Día laborable"); ax.set_ylabel("Nº de pacientes")
        ax.set_title(f"Pacientes citados por día · {stem}")
        self._save(fig, f"B18_pacientes_por_dia_{stem}")

    # ════════════════════════════════════════════════════════════════════════
    # GRUPO C — DATOS CRUDOS (.dat): demanda, MCP, déficit
    # ════════════════════════════════════════════════════════════════════════
    def _matriz_demanda(self):
        dats = [(s, d) for s, d in self._all_dats() if _stem_num(s) != 14]
        M = np.zeros((7, len(dats)))
        stems = []
        for j, (stem, dat) in enumerate(dats):
            stems.append(stem.replace("istanza", ""))
            dem = Counter()
            for p in dat["patients"].values():
                dem[p["alpha"]] += p["v"]
            for k in range(1, 8):
                M[k - 1, j] = dem[k]
        return M, stems, dats

    def fig_C19_heatmap_demanda(self):
        M, stems, _ = self._matriz_demanda()
        fig, ax = plt.subplots(figsize=(11, 3.2))
        im = ax.imshow(M, aspect="auto", cmap="magma")
        ax.set_yticks(range(7)); ax.set_yticklabels([PATOLOGIAS[k] for k in range(1, 8)])
        ax.set_xticks(range(len(stems))); ax.set_xticklabels(stems, rotation=90, fontsize=6)
        ax.set_xlabel("Instancia real (semana)")
        ax.set_title("Demanda de visita por patología y semana (slots)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="slots de visita")
        self._save(fig, "C19_heatmap_demanda")

    def _matriz_firma(self):
        dats = self._all_dats()
        M = np.zeros((7, len(dats)))
        stems = []
        for j, (stem, dat) in enumerate(dats):
            stems.append(stem.replace("istanza", ""))
            firma = Counter()
            for (_r, k, _d) in dat["w"]:
                firma[k] += 1
            for k in range(1, 8):
                M[k - 1, j] = firma[k]
        return M, stems, dats

    def fig_C20_heatmap_firma_mcp(self):
        M, stems, dats = self._matriz_firma()
        fig, ax = plt.subplots(figsize=(11, 3.2))
        im = ax.imshow(M, aspect="auto", cmap="YlGnBu")
        ax.set_yticks(range(7)); ax.set_yticklabels([PATOLOGIAS[k] for k in range(1, 8)])
        ax.set_xticks(range(len(stems))); ax.set_xticklabels(stems, rotation=90, fontsize=6)
        # líneas verticales donde cambia la firma (= cambio de mes)
        prev = None
        for j, (_stem, dat) in enumerate(dats):
            firma = tuple(Counter(k for (_r, k, _d) in dat["w"])[k] for k in range(1, 8))
            firma = tuple(Counter([k for (_r, k, _d) in dat["w"]])[k] for k in range(1, 8))
            if prev is not None and firma != prev:
                ax.axvline(j - 0.5, color="red", lw=1.3)
            prev = firma
        for k in range(1, 8):
            for j in range(len(stems)):
                ax.text(j, k - 1, int(M[k - 1, j]), ha="center", va="center", fontsize=5)
        ax.set_xlabel("Instancia real (semana)")
        ax.set_title("Firma del MCP (bloques/patología): las bandas constantes = los 10 meses")
        ax.grid(False)
        self._save(fig, "C20_heatmap_firma_mcp")

    def fig_C21_heatmap_deficit(self):
        """Déficit del MCP crudo: dem - 36*salas (solo positivos = overtime forzado)."""
        dats = self._all_dats()
        M = np.full((7, len(dats)), np.nan)
        stems = []
        for j, (stem, dat) in enumerate(dats):
            stems.append(stem.replace("istanza", ""))
            g = dat["giornilav"]
            c_map = dat["c_map"] or {i: i for i in range(1, g + 1)}
            salas = {k: 0 for k in range(1, 8)}
            for lab in range(1, g + 1):
                dnat = c_map[lab]
                for (_r, k, dd) in dat["w"]:
                    if dd == dnat:
                        salas[k] += 1
            dem = Counter()
            for p in dat["patients"].values():
                dem[p["alpha"]] += p["v"]
            for k in range(1, 8):
                defi = dem[k] - NV * salas[k]
                M[k - 1, j] = defi if defi > 0 else np.nan
        fig, ax = plt.subplots(figsize=(11, 3.2))
        cmap = plt.cm.Reds.copy(); cmap.set_bad("#eeeeee")
        im = ax.imshow(M, aspect="auto", cmap=cmap)
        ax.set_yticks(range(7)); ax.set_yticklabels([PATOLOGIAS[k] for k in range(1, 8)])
        ax.set_xticks(range(len(stems))); ax.set_xticklabels(stems, rotation=90, fontsize=6)
        ax.set_xlabel("Instancia real (semana)")
        ax.set_title("Déficit estructural del MCP CRUDO (gris = sin déficit)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="slots en déficit (overtime forzado)")
        self._save(fig, "C21_heatmap_deficit_crudo")

    def fig_C22_pacientes_semana(self):
        df = pd.DataFrame([(s, sum(1 for _ in d["patients"]),
                            sum(1 for p in d["patients"].values() if p["lam"] == 1))
                           for s, d in self._all_dats() if _stem_num(s) != 14],
                          columns=["inst", "ptot", "crit"])
        x = np.arange(len(df))
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(x, df["ptot"], "-o", color=COL_C, ms=3, label="Pacientes totales")
        ax.plot(x, df["crit"], "-o", color=COL_ART, ms=3, label="Críticos")
        ax2 = ax.twinx()
        ax2.plot(x, 100 * df["crit"] / df["ptot"], color=COL_BAD, lw=1, ls=":",
                 label="% críticos")
        ax2.set_ylabel("% críticos", color=COL_BAD); ax2.set_ylim(0, 40); ax2.grid(False)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("istanza", "") for s in df["inst"]], rotation=90, fontsize=6)
        ax.set_xlabel("Instancia real (semana)"); ax.set_ylabel("Nº de pacientes")
        ax.set_title("Volumen de pacientes y fracción crítica por semana")
        ax.legend(loc="upper left")
        self._save(fig, "C22_pacientes_semana")

    def fig_C23_mezcla_patologias(self):
        M, stems, _ = self._matriz_demanda()
        prop = M / M.sum(axis=0, keepdims=True) * 100
        x = np.arange(len(stems))
        fig, ax = plt.subplots(figsize=(11, 3.6))
        base = np.zeros(len(stems))
        for k in range(7):
            ax.bar(x, prop[k], bottom=base, color=OKABE[k % len(OKABE)],
                   label=PATOLOGIAS[k + 1], width=1.0)
            base += prop[k]
        ax.set_xticks(x); ax.set_xticklabels(stems, rotation=90, fontsize=6)
        ax.set_xlabel("Instancia real (semana)"); ax.set_ylabel("% de la demanda")
        ax.set_title("Mezcla de patologías por semana (demanda de visita)")
        ax.set_ylim(0, 100)
        ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=7)
        self._save(fig, "C23_mezcla_patologias")

    def fig_C24_hist_pacientes(self):
        ptot = [sum(1 for _ in d["patients"]) for s, d in self._all_dats() if _stem_num(s) != 14]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(ptot, bins=12, color=COL_C, alpha=0.85, edgecolor="white")
        ax.axvline(np.mean(ptot), color=COL_BAD, ls="--", label=f"media {np.mean(ptot):.0f}")
        ax.set_xlabel("Pacientes por semana"); ax.set_ylabel("Nº de instancias")
        ax.set_title("Distribución del volumen semanal de pacientes")
        ax.legend()
        self._save(fig, "C24_hist_pacientes")

    def fig_C25_hist_infusion(self):
        fs = []
        for s, d in self._all_dats():
            if _stem_num(s) == 14:
                continue
            fs += [p["f"] for p in d["patients"].values()]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(fs, bins=np.arange(min(fs), max(fs) + 2) - 0.5, color=COL_ART,
                alpha=0.85, edgecolor="white")
        ax.set_xlabel("Duración de la infusión (slots de 10 min)")
        ax.set_ylabel("Nº de pacientes (todas las semanas)")
        ax.set_title("Distribución de la duración de las infusiones")
        sec = ax.secondary_xaxis("top", functions=(lambda s: s * 10 / 60, lambda h: h * 60 / 10))
        sec.set_xlabel("horas")
        self._save(fig, "C25_hist_infusion")

    def fig_C26_utilizacion_vs_tabla8(self):
        """Utilización por patología: Opción C (calculada) vs Tabla 8 del artículo."""
        salasC = self._opcionC_salas_por_stem()
        util = {k: [] for k in range(1, 8)}
        for stem, dat in self._all_dats():
            if _stem_num(stem) == 14:
                continue
            dem = Counter()
            for p in dat["patients"].values():
                dem[p["alpha"]] += p["v"]
            sk = salasC[stem]
            for k in range(1, 8):
                if sk[k] > 0:
                    util[k].append(100 * dem[k] / (NV * sk[k]))
        pats = [PATOLOGIAS[k] for k in range(1, 8)]
        nuestro_avg = [np.mean(util[k]) for k in range(1, 8)]
        art_avg = [ART_TABLA8[p][0] for p in pats]
        x = np.arange(7); w = 0.38
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - w / 2, nuestro_avg, w, color=COL_C, label="Opción C (nuestro)")
        ax.bar(x + w / 2, art_avg, w, color=COL_ART, label="Artículo (Tabla 8)")
        glob_n = np.mean([v for k in range(1, 8) for v in util[k]])
        ax.axhline(glob_n, color=COL_C, ls=":", lw=1)
        ax.axhline(ART_TABLA8["Global"][0], color=COL_ART, ls=":", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(pats)
        ax.set_ylabel("Utilización media de salas (%)")
        ax.set_title(f"Utilización por patología: global nuestro {glob_n:.1f}% vs artículo {ART_TABLA8['Global'][0]}%")
        ax.legend()
        self._save(fig, "C26_utilizacion_vs_tabla8")

    # ════════════════════════════════════════════════════════════════════════
    # GRUPO D — DATOS SIMULADOS (90 runs)
    # ════════════════════════════════════════════════════════════════════════
    def fig_D28_validacion_sim(self):
        df = self._resumen_sim()
        fig, ax = plt.subplots(figsize=(6, 4))
        n_f1 = (df["f1_overtime_slots"] == 0).sum()
        n_f2 = (df["f2_maxwait_slots"] == 0).sum()
        b = ax.bar(["$F_1=0$", "$F_2=0$", "solved $P_1/P_2/P_3$"],
                   [n_f1, n_f2, (df["status_p3"] == "solved").sum()],
                   color=OKABE[:3], alpha=0.85)
        for rect, v in zip(b, [n_f1, n_f2, (df["status_p3"] == "solved").sum()]):
            ax.annotate(f"{v}/{len(df)}", (rect.get_x() + rect.get_width() / 2, v),
                        ha="center", va="bottom")
        ax.set_title("Validación sobre datos simulados (90 ejecuciones)")
        ax.set_ylabel("Ejecuciones"); ax.set_ylim(0, len(df) * 1.12)
        self._save(fig, "D28_validacion_sim")

    def fig_D29_sillon_por_fraccion(self):
        df = self._resumen_sim()
        fig, ax = plt.subplots(figsize=(6, 4))
        orden = ["1of3", "2of3", "3of3"]
        data = [df[df["fraccion"] == fr]["pct_sillon"] for fr in orden]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=["1/3", "2/3", "3/3"], showmeans=True)
        for patch, c in zip(bp["boxes"], OKABE[:3]):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_ylabel("% pacientes en sillón")
        ax.set_xlabel("Fracción de pacientes")
        ax.set_title("% de sillón por fracción (datos simulados)")
        self._save(fig, "D29_sillon_por_fraccion")

    def fig_D30_tiempo_vs_pacientes_sim(self):
        df = self._resumen_sim()
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for fr, c in zip(["1of3", "2of3", "3of3"], OKABE[:3]):
            sub = df[df["fraccion"] == fr]
            ax.scatter(sub["n_pacientes"], sub["t_total_s"], color=c, s=26,
                       label=fr.replace("of3", "/3"), alpha=0.8, edgecolor="white", lw=0.4)
        ax.set_xlabel("Nº de pacientes"); ax.set_ylabel("Tiempo total (s)")
        ax.set_title("Escalabilidad con Gurobi (datos simulados)")
        ax.legend(title="Fracción")
        self._save(fig, "D30_tiempo_vs_pacientes_sim")

    def fig_D31_escalabilidad(self):
        df = self._resumen_sim()
        orden = ["1of3", "2of3", "3of3"]
        fig, ax = plt.subplots(figsize=(6, 4))
        data = [df[df["fraccion"] == fr]["t_total_s"] for fr in orden]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=["1/3", "2/3", "3/3"], showmeans=True)
        for patch, c in zip(bp["boxes"], OKABE[:3]):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_ylabel("Tiempo total (s)"); ax.set_xlabel("Fracción de pacientes")
        ax.set_title("Crecimiento del tiempo de resolución con el tamaño")
        self._save(fig, "D31_escalabilidad")

    # ════════════════════════════════════════════════════════════════════════
    # GRUPO F — CRUDO vs OPCIÓN C  +  interioridades del solver (logs)
    # ════════════════════════════════════════════════════════════════════════
    def fig_F1_overtime_crudo_vs_c(self):
        """Overtime crudo (de los logs/CSV crudo disponibles) vs Opción C (resumen real).
        El barrido crudo puede ser parcial: se grafican solo las instancias con dato crudo."""
        dc = self._resumen_real().set_index("instancia")["f1_overtime_slots"]
        rc = self._resumen_crudo()
        crudo_f1 = {}
        if not rc.empty:  # preferir el CSV resumen crudo si existe
            for _, r in rc.iterrows():
                crudo_f1[r["instancia"]] = r["f1_overtime_slots"]
        else:             # si no, reconstruir desde los logs crudos disponibles
            for p in sorted(glob(os.path.join(self.DIR_LOGS_CRUDO, "run_*.txt")), key=_stem_num):
                stem = os.path.basename(p).replace("run_", "").replace(".txt", "")
                lg = self._parse_log(stem, crudo=True)
                if not np.isnan(lg["f1"]):
                    crudo_f1[stem] = lg["f1"]
        comunes = [s for s in crudo_f1 if s in dc.index]
        comunes.sort(key=_stem_num)
        if not comunes:
            print("  [SKIP] F1_overtime_crudo_vs_c: sin datos de MCP crudo disponibles")
            return
        x = np.arange(len(comunes)); w = 0.42
        fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(comunes) + 2), 3.8))
        ax.bar(x - w / 2, [crudo_f1[s] for s in comunes], w, color=COL_CRUDO, label="MCP crudo")
        ax.bar(x + w / 2, [dc[s] for s in comunes], w, color=COL_C, label="Opción C")
        for i, s in enumerate(comunes):
            ax.annotate(f"{crudo_f1[s]:.0f}", (i - w / 2, crudo_f1[s]), ha="center",
                        va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("istanza", "") for s in comunes])
        ax.set_xlabel("Instancia real"); ax.set_ylabel("Overtime $F_1$ (slots)")
        ax.set_title("Overtime forzado por el MCP crudo vs eliminado por la Opción C")
        ax.legend()
        note = "(barrido crudo parcial)" if len(comunes) < 40 else ""
        if note:
            ax.text(0.99, 0.95, note, transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, style="italic", color="gray")
        self._save(fig, "F1_overtime_crudo_vs_c")

    def fig_F2_lb1_vs_conteo(self):
        """Sobre los logs del MCP crudo: cota válida LB1 vs cota de conteo (déficit)."""
        rows = []
        for stem, _ in self._all_dats():
            if _stem_num(stem) == 14:
                continue
            lg = self._parse_log(stem, crudo=True)
            if not np.isnan(lg["conteo"]) and lg["conteo"] > 0:
                rows.append((stem, lg["lb1"], lg["conteo"], lg["f1"]))
        if not rows:
            return
        d = pd.DataFrame(rows, columns=["inst", "lb1", "conteo", "f1"]).sort_values(
            "conteo", ascending=False)
        x = np.arange(len(d)); w = 0.27
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - w, d["conteo"], w, color=COL_CRUDO, label="conteo (déficit, INVÁLIDA)")
        ax.bar(x, d["lb1"], w, color=COL_C, label="LB1 reforzada (VÁLIDA)")
        ax.bar(x + w, d["f1"], w, color=COL_OK, label="$F_1$ real")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("istanza", "") for s in d["inst"]], rotation=90, fontsize=6)
        ax.set_ylabel("Slots de overtime")
        ax.set_title("MCP crudo: la cota de conteo (déficit) sobreestima el overtime real")
        ax.legend()
        self._save(fig, "F2_lb1_vs_conteo_crudo")

    def fig_F3_nodos_p1(self):
        """Nodos de branch&bound de P1: Opción C (cierra en la raíz) vs MCP crudo."""
        rows = []
        for stem, _ in self._all_dats():
            if _stem_num(stem) == 14:
                continue
            lc = self._parse_log(stem, crudo=False)
            lk = self._parse_log(stem, crudo=True)
            rows.append((stem, lc["p1_nodes"], lk["p1_nodes"]))
        d = pd.DataFrame(rows, columns=["inst", "C", "crudo"]).dropna(how="all",
                                                                      subset=["C", "crudo"])
        x = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.bar(x - 0.2, d["crudo"].fillna(0), 0.4, color=COL_CRUDO, label="MCP crudo")
        ax.bar(x + 0.2, d["C"].fillna(0), 0.4, color=COL_C, label="Opción C")
        ax.set_yscale("symlog")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("istanza", "") for s in d["inst"]], rotation=90, fontsize=6)
        ax.set_ylabel("Nodos B&B en $P_1$ (escala log)")
        ax.set_title("Coste de $P_1$: la Opción C cierra en la raíz; el MCP crudo ramifica")
        ax.legend()
        self._save(fig, "F3_nodos_p1")

    def fig_F4_gap_p3(self):
        rows = []
        for stem, _ in self._all_dats():
            if _stem_num(stem) == 14:
                continue
            lg = self._parse_log(stem, crudo=False)
            rows.append((stem, lg["p3_gap"]))
        d = pd.DataFrame(rows, columns=["inst", "gap"]).dropna()
        x = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(10, 3.6))
        col = [COL_BAD if g > 5 else COL_C for g in d["gap"]]
        ax.bar(x, d["gap"], color=col)
        ax.axhline(5, color="gray", ls="--", lw=1, label="objetivo 5%")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("istanza", "") for s in d["inst"]], rotation=90, fontsize=6)
        ax.set_ylabel("Gap final de $P_3$ (%)")
        ax.set_title("Gap de optimización de $P_3$ por instancia (Opción C)")
        ax.legend()
        self._save(fig, "F4_gap_p3")

    # ════════════════════════════════════════════════════════════════════════
    # GRUPO E — ESQUEMAS CONCEPTUALES / MAPAS DE IDEAS (matplotlib puro)
    # ════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _caja(ax, xy, w, h, texto, color, fc=None, fs=9, tc="black"):
        x, y = xy
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                             linewidth=1.4, edgecolor=color, facecolor=fc or color + "33")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs, color=tc)
        return (x + w / 2, y + h / 2)

    @staticmethod
    def _flecha(ax, p0, p1, color="black", text=None):
        ar = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                             lw=1.4, color=color, shrinkA=6, shrinkB=6)
        ax.add_patch(ar)
        if text:
            mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            ax.text(mx, my + 0.03, text, ha="center", va="bottom", fontsize=8, color=color)

    def fig_E33_niveles_decision(self):
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
        niveles = [("ESTRATÉGICO\n(años)", "Dimensionar infraestructura:\nnº de salas, sillones, camas", OKABE[3], 4.3),
                   ("TÁCTICO\n(meses)", "MCP: qué salas se dedican a\ncada patología cada día", OKABE[1], 2.6),
                   ("OPERATIVO\n(semana/día)", "Citación: qué día y a qué hora\nse atiende a cada paciente", OKABE[0], 0.9)]
        for txt, desc, c, y in niveles:
            self._caja(ax, (0.4, y), 2.6, 1.3, txt, c, fs=9)
            self._caja(ax, (3.3, y), 6.2, 1.3, desc, c, fc="white", fs=9)
        self._flecha(ax, (1.7, 4.3), (1.7, 3.9), color="gray")
        self._flecha(ax, (1.7, 2.6), (1.7, 2.2), color="gray")
        ax.text(5, 5.7, "Niveles de decisión en planificación hospitalaria",
                ha="center", fontsize=11, fontweight="bold")
        ax.text(9.4, 0.3, "← el modelo del TFG actúa AQUÍ (toma el MCP como dato fijo)",
                ha="right", fontsize=8, style="italic", color=OKABE[0])
        self._save(fig, "E33_niveles_decision")

    def fig_E34_flujo_paciente(self):
        fig, ax = plt.subplots(figsize=(9, 2.8))
        ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
        etapas = [("Llegada", OKABE[7]), ("Analítica\n+ validación", OKABE[5]),
                  ("VISITA\n(sala, 10-20 min)", OKABE[0]), ("Farmacia", OKABE[6]),
                  ("INFUSIÓN\n(sillón o cama)", OKABE[2])]
        xs = np.linspace(0.3, 9.5, len(etapas))
        centros = []
        for (txt, c), x in zip(etapas, xs):
            centros.append(self._caja(ax, (x, 1.0), 1.7, 1.1, txt, c, fc="white", fs=8))
        for i in range(len(centros) - 1):
            self._flecha(ax, (xs[i] + 1.7, 1.55), (xs[i + 1], 1.55))
        ax.text(6, 2.7, "Flujo del paciente — política «mismo día» (same-day)",
                ha="center", fontsize=11, fontweight="bold")
        ax.text(xs[2] + 0.85, 0.7, "$F_1$: overtime", ha="center", fontsize=8, color=OKABE[0])
        ax.text((xs[2] + xs[4]) / 2 + 0.85, 0.55, "$F_2$: espera entre etapas",
                ha="center", fontsize=8, color="gray")
        ax.text(xs[4] + 0.85, 0.7, "$F_3$: sillón (confort)", ha="center", fontsize=8, color=OKABE[2])
        self._save(fig, "E34_flujo_paciente")

    def fig_E35_pipeline_lexicografico(self):
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.set_xlim(0, 12); ax.set_ylim(0, 3.2); ax.axis("off")
        p1 = self._caja(ax, (0.3, 1.1), 3.0, 1.2,
                        "$P_1$: min $F_1$\n(overtime)", OKABE[0], fc="white")
        p2 = self._caja(ax, (4.4, 1.1), 3.0, 1.2,
                        "$P_2$: min $F_2$ (espera)\ns.a. $F_1 \\leq \\bar v_1$", OKABE[1], fc="white")
        p3 = self._caja(ax, (8.5, 1.1), 3.2, 1.2,
                        "$P_3$: max $F_3$ (sillón)\ns.a. $F_1\\leq\\bar v_1, F_2\\leq\\bar v_2$",
                        OKABE[2], fc="white")
        self._flecha(ax, (3.3, 1.7), (4.4, 1.7), text="$\\bar v_1$")
        self._flecha(ax, (7.4, 1.7), (8.5, 1.7), text="$\\bar v_2$")
        ax.text(6, 2.9, "Optimización lexicográfica: $F_1 \\succ F_2 \\succ F_3$",
                ha="center", fontsize=11, fontweight="bold")
        ax.text(6, 0.5, "cada etapa congela el óptimo de la anterior (ε-constraint)",
                ha="center", fontsize=8, style="italic")
        self._save(fig, "E35_pipeline_lexicografico")

    def fig_E36_ventana_deslizante(self):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_xlim(0, 12); ax.set_ylim(0, 4)
        for s in range(12):
            ax.add_patch(Rectangle((s, 2), 1, 1, fill=False, edgecolor="gray"))
            ax.text(s + 0.5, 1.7, f"s={s+1}", ha="center", fontsize=7, color="gray")
        # un paciente con visita de v=3 que empieza en s=4
        for s in range(3, 6):
            ax.add_patch(Rectangle((s, 2), 1, 1, color=OKABE[0], alpha=0.6))
        ax.text(4.5, 3.4, "visita de un paciente ($v_p=3$ slots)", ha="center", fontsize=8,
                color=OKABE[0])
        ax.annotate("", (3, 1.4), (6, 1.4), arrowprops=dict(arrowstyle="<->", color=COL_BAD))
        ax.text(4.5, 1.1, "en el slot s=5 hay 1 visita 'en curso' de este paciente\n"
                          "(ventana [s-v+1, s])", ha="center", fontsize=8, color=COL_BAD)
        ax.set_title("Restricción de capacidad por ventana deslizante")
        ax.axis("off")
        self._save(fig, "E36_ventana_deslizante")

    def fig_E37_mcp_rejilla(self, stem="istanza03"):
        dat = self._parse_dat(os.path.join(self.DIR_REAL, f"{stem}.dat"))
        g = 5
        grid = np.zeros((6, 5), dtype=int)  # salas x dias
        for (r, k, d) in dat["w"]:
            if 1 <= r <= 6 and 1 <= d <= 5:
                grid[r - 1, d - 1] = k
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        cmap = matplotlib.colors.ListedColormap(["white"] + OKABE[:7])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=7, aspect="auto")
        for r in range(6):
            for d in range(5):
                k = grid[r, d]
                if k > 0:
                    ax.text(d, r, PATOLOGIAS[k], ha="center", va="center", fontsize=9,
                            color="white", fontweight="bold")
        ax.set_xticks(range(5)); ax.set_xticklabels([f"Día {d}" for d in range(1, 6)])
        ax.set_yticks(range(6)); ax.set_yticklabels([f"Sala {r}" for r in range(1, 7)])
        ax.set_title(f"MCP como rejilla salas × días (block scheduling) · {stem}")
        ax.grid(False)
        self._save(fig, f"E37_mcp_rejilla_{stem}")

    def fig_E38_palomar(self):
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
        ax.text(6, 4.6, "Principio del palomar: si demanda > capacidad regular → overtime forzado",
                ha="center", fontsize=10, fontweight="bold")
        # capacidad
        for i in range(9):
            ax.add_patch(Rectangle((0.3 + i * 0.6, 2.6), 0.5, 0.9, color=OKABE[2], alpha=0.5))
        ax.text(3, 3.7, "9 bloques de HE × 36 = 324 slots regulares", ha="center", fontsize=8,
                color=OKABE[2])
        # demanda
        for i in range(11):
            ax.add_patch(Rectangle((0.3 + i * 0.6, 1.0), 0.5, 0.9,
                                   color=OKABE[0] if i < 9 else COL_BAD, alpha=0.6))
        ax.text(3.3, 0.5, "demanda HE = 370 slots → 46 NO caben (en rojo)", ha="center",
                fontsize=8, color=COL_BAD)
        ax.text(9.2, 1.9, "déficit 46 = capacidad que falta\n"
                          "(≠ overtime: las salas\ndesbordan en paralelo →\no. real ≈ 46/3 ≈ 15)",
                ha="left", fontsize=8, style="italic")
        self._save(fig, "E38_palomar")

    def fig_E39_mapa_mental(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
        c = self._caja(ax, (4.0, 3.6), 2.0, 0.9, "TFG\nCitación oncológica", OKABE[7],
                       fc=OKABE[7] + "22", fs=9)
        ramas = [
            ("Fase 1\nSintéticos + CBC", OKABE[5], (0.4, 6.5)),
            ("Fase 2\nAMPL+Gurobi (90 runs)", OKABE[4], (0.4, 4.0)),
            ("Fase 3-5\nMCP, festivos, LB1", OKABE[1], (0.4, 1.4)),
            ("Fase 6\nDiagnóstico fidelidad", OKABE[3], (7.6, 6.5)),
            ("Decisión: Opción C\n(MCP mensual)", OKABE[0], (7.6, 4.0)),
            ("Fase 7\nBarrido 51 + realismo", OKABE[2], (7.6, 1.4)),
        ]
        for txt, col, xy in ramas:
            cc = self._caja(ax, xy, 2.2, 1.0, txt, col, fc="white", fs=8)
            self._flecha(ax, (5.0, 4.05), cc, color="gray")
        ax.text(5, 7.6, "Mapa de desarrollo del proyecto", ha="center", fontsize=12,
                fontweight="bold")
        self._save(fig, "E39_mapa_mental")

    def fig_E40_crudo_vs_c_esquema(self):
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
        ax.text(6, 3.7, "MCP crudo vs Opción C (re-balanceo mensual)", ha="center",
                fontsize=11, fontweight="bold")
        self._caja(ax, (0.4, 1.8), 3.2, 1.2,
                   "MCP CRUDO (.dat)\nHE infra-dotada → déficit\nistanza03: F1≈18, P1 no cierra",
                   COL_CRUDO, fc="white", fs=8)
        self._caja(ax, (4.4, 1.8), 3.0, 1.2,
                   "Re-balanceo al pico\nmensual (30 bloques)\n método del artículo",
                   OKABE[1], fc="white", fs=8)
        self._caja(ax, (8.2, 1.8), 3.4, 1.2,
                   "OPCIÓN C\nsin déficit (5 días)\nF1=0, P1 en 1 nodo",
                   COL_C, fc="white", fs=8)
        self._flecha(ax, (3.6, 2.4), (4.4, 2.4))
        self._flecha(ax, (7.4, 2.4), (8.2, 2.4))
        ax.text(6, 0.8, "respeta la fijeza mensual del MCP (no relaja restricciones reales)",
                ha="center", fontsize=8, style="italic")
        self._save(fig, "E40_crudo_vs_c_esquema")

    # ════════════════════════════════════════════════════════════════════════
    def generar_todo(self):
        metodos = [m for m in dir(self) if m.startswith("fig_")]
        # orden por grupo (A,B,C,D,F,E) y número
        def clave(m):
            grupo = m.split("_")[1][0]
            num = re.search(r"(\d+)", m.split("_")[1])
            return (grupo, int(num.group(1)) if num else 0)
        metodos.sort(key=clave)
        print(f"Generando {len(metodos)} figuras en {self.outdir}/ ...")
        ok, err = 0, 0
        for m in metodos:
            try:
                getattr(self, m)()
                ok += 1
            except Exception as e:
                err += 1
                print(f"  [ERROR] {m}: {type(e).__name__}: {e}")
        print(f"\nHecho: {ok} figuras OK, {err} con error. Carpeta: {self.outdir}/")


if __name__ == "__main__":
    GeneradorFiguras().generar_todo()
