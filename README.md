# IICAS-Data-Sim — Optimización y simulación de flujo de pacientes en un hospital de día oncológico

Trabajo Fin de Grado (Grado en Ingeniería Informática, UCLM).

El proyecto implementa un **modelo MILP multiobjetivo lexicográfico** ($P_1\rightarrow P_2\rightarrow P_3$)
que decide, para cada paciente y cada semana, **qué día** y **a qué hora** se realizan su **visita**
(consulta con oncólogo, consume una sala) y su **infusión** (consume un sillón o una cama), con tres
objetivos por orden de prioridad:

| Prioridad | Objetivo | Descripción |
|---|---|---|
| **P1** | $\min F_1$ | minimizar el **overtime** (horas extra del personal) |
| **P2** | $\min F_2$ | minimizar la **espera** del paciente entre visita e infusión ($F_1 \le \bar v_1$) |
| **P3** | $\max F_3$ | maximizar el **confort** (nº de no críticos en sillón) ($F_1 \le \bar v_1,\ F_2 \le \bar v_2$) |

---

## Tabla de contenidos
1. [Estado del proyecto](#estado-del-proyecto)
2. [Requisitos y versiones exactas](#requisitos-y-versiones-exactas)
3. [Puesta en marcha del entorno](#puesta-en-marcha-del-entorno)
4. [Estructura del proyecto (árbol)](#estructura-del-proyecto-árbol)
5. [Qué es y para qué sirve cada `.py`](#qué-es-y-para-qué-sirve-cada-py)
6. [Carpetas de datos, resultados y logs](#carpetas-de-datos-resultados-y-logs)
7. [Guía de reproducibilidad paso a paso](#guía-de-reproducibilidad-paso-a-paso)
8. [La semilla y la generación de datos sintéticos](#la-semilla-y-la-generación-de-datos-sintéticos)
9. [Parámetros del modelo (constantes)](#parámetros-del-modelo-constantes)
10. [Formato de los datos](#formato-de-los-datos)
11. [Documentación, figuras y pseudocódigo](#documentación-figuras-y-pseudocódigo)
12. [Notas de reproducibilidad y limitaciones](#notas-de-reproducibilidad-y-limitaciones)
13. [Autoría y referencia](#autoría-y-referencia)

---

## Estado del proyecto

- **Modelo definitivo:** `model-solver-real.py` (AMPL + Gurobi) con la **Opción C** (MCP mensual
  re-balanceado), la **cota LB1 reforzada** y *warm starts*.
- **Barridos completados:**
  - **Datos simulados:** 30 instancias × 3 fracciones = **90 ejecuciones** (validación de escala).
  - **Datos reales (Opción C):** **51 instancias** → overtime 0 en 49/51, espera 0 en 50/51.
  - **Datos reales (MCP crudo):** **51 instancias** (diagnóstico: 4 infactibles, overtime forzado
    en 25/47).
- **Figuras:** 41 (PDF + PNG) generadas por `generador_figuras.py`.
- **Memoria del TFG:** `docs/tfg_draft_6.pdf` (borrador vigente, 77 pp).

---

## Requisitos y versiones exactas

Versiones **verificadas** de este entorno (algunas ejecuciones se hicieron en otra máquina — ver
[limitaciones](#notas-de-reproducibilidad-y-limitaciones)):

| Componente | Versión | Notas |
|---|---|---|
| **Python** | **3.11.15** | del `venv` del proyecto |
| **amplpy** | **0.16.1** | interfaz Python ↔ AMPL |
| **AMPL** | **20260520** | motor de modelado (licencia académica) |
| **Gurobi** | **13.0.2** (reales, macOS ARM) / **13.0.1** (simulados, Windows 11) | motor de optimización, vía módulo `gurobi` de amplpy |
| **pandas** | 3.0.3 | lectura de CSV / análisis |
| **numpy** | 2.4.6 | generación de datos y cálculos |
| **matplotlib** | 3.11.0 | figuras (`generador_figuras.py`) |
| **pypdf** | (opcional) | extracción de texto del PDF del borrador |
| **TinyTeX / TeX Live 2026** | (opcional) | compilar el anexo de pseudocódigo `docs/pseudocodigo/` |

> **Licencia AMPL/Gurobi.** Cada solver de datos reales requiere activar el módulo con un UUID de
> licencia AMPL: variable `AMPL_UUID` al principio de `model-solver-real*.py`. Es una licencia
> académica temporal la usada para este TFG; **debes poner la tuya si dispones de una para poder reproducir los resultados**.

---

## Puesta en marcha del entorno

```bash
# 1) Crear y activar el entorno virtual
python3.11 -m venv venv
source venv/bin/activate            # (Windows: venv\Scripts\activate)

# 2) Dependencias de Python
pip install amplpy pandas numpy matplotlib pypdf

# 3) Instalar los módulos de AMPL (incluye Gurobi)
python -m amplpy.modules install gurobi        # + base (se instala solo)

# 4) Activar la licencia AMPL (pon tu UUID en AMPL_UUID dentro de los .py,
#    o actívala globalmente):
python -m amplpy.modules activate <TU-UUID-AMPL>
```

Comprobación rápida:
```bash
python -c "import amplpy, pandas, numpy, matplotlib; from amplpy import AMPL; print('OK', AMPL().get_option('version'))"
```

---

## Estructura del proyecto (árbol)

```
IICAS-Data-Sim/
├── dataset-generator.py            # Genera 30 instancias sintéticas (semilla 20260427)
├── model-solver.py                 # Fase 1  — MILP lexicográfico en PuLP/CBC (sintéticos, cap. global)
├── model-solver-ampl.py            # Fase 2  — mismo modelo en AMPL/Gurobi (sintéticos)
├── model-solver-real.py            # PRINCIPAL — AMPL/Gurobi, datos reales, MCP + Opción C + LB1 + festivos
├── model-solver-real-mcp-crudo.py  # Variante — MCP crudo (sin re-balanceo), para el diagnóstico
├── model-solver-real-debug.py      # Depuración de instancias concretas (11/17/18) con log completo
├── model-solver-toy.py             # Toy de verificación (1 día, 7 pacientes)
├── model-solver-toy-peor.py        # Toy "peor caso" (3 días, estresa P1 y LB1)
├── generador_figuras.py            # Clase GeneradorFiguras: 41 figuras del TFG (PDF+PNG)
│
├── Databases/
│   ├── real-sets/                  # 52 .dat reales del Hospital San Martino (51 usadas; istanza14 excluida)
│   ├── simulated-sets/             # 30 .csv sintéticos (salida de dataset-generator.py)
│   ├── optimized-sets/             # Resultados simulados: 90 agendas + resumen_ejecucion_ampl.csv
│   ├── optimized-sets-real/        # Resultados Opción C: 51 agendas + resumen_ejecucion_real.csv
│   ├── optimized-sets-real-crudo/  # Resultados MCP crudo: 47 agendas + resumen_ejecucion_real_crudo.csv
│   └── optimized-sets-real-debug/  # Resultados de las instancias de control (11/17/18)
│
├── logs/                           # Logs de la fase sintética (PuLP/CBC)
├── logs-ampl/                      # Logs de las 90 ejecuciones simuladas (AMPL/Gurobi)
├── logs-real/                      # 51 logs de Gurobi del barrido real (Opción C)
├── logs-real-crudo/                # 51 logs de Gurobi del barrido con MCP crudo
├── logs-real-debug/                # Logs completos de las instancias de control
│
├── figuras/                        # 41 figuras (PDF + PNG) para la memoria
│
├── venv/                           # Entorno virtual de Python (no versionado)
│
└── README.md                       # (este fichero)
```

> **Nota sobre control de versiones.** El `.gitignore` excluye datos, logs, `docs/`, `figuras/`,
> `mds-to-word/`, `venv/`, backups y otros datos de depuración. Es decir, Git rastrea esencialmente los
> **scripts `.py`**; el resto es material generado o documentación local.
---

## Qué es y para qué sirve cada `.py`

| Fichero | Fase | Solver | Datos | Descripción |
|---|---|---|---|---|
| **`dataset-generator.py`** | 1 | — | genera | Crea **30 instancias sintéticas** que replican el perfil del hospital (patologías, criticidad 24,46 %, duraciones). Salida en `Databases/simulated-sets/`. Reproducible por semilla (§8). |
| **`model-solver.py`** | 1 | **PuLP + CBC** | sintéticos | Primer prototipo del MILP lexicográfico. Capacidad de salas **global** ($\sum x \le 6$, sin MCP). Incluye la **heurística de fracciones progresivas** (1/3, 2/3, 3/3). CBC agota memoria (OOM) en 2/3 y 3/3 → motiva el salto a Gurobi. |
| **`model-solver-ampl.py`** | 2 | **AMPL + Gurobi** | sintéticos | Reimplementación del **mismo modelo** en AMPL/Gurobi. Re-ejecuta el banco completo (**90 ejecuciones** = 30 × 3 fracciones); resuelve la escala que CBC no podía. Salida en `Databases/optimized-sets/` y `logs-ampl/`. |
| **`model-solver-real.py`** | 3–7 | **AMPL + Gurobi** | reales `.dat` | **SCRIPT PRINCIPAL / DEFINITIVO.** Añade la **restricción MCP por patología** (ecuación 2 del artículo), el tratamiento de **días festivos**, la **cota inferior válida LB1 reforzada**, los *warm starts* por descomposición diaria (Procedimiento 2) y la **reconstrucción mensual del MCP (Opción C)**. Salida en `Databases/optimized-sets-real/` y `logs-real/`. |
| **`model-solver-real-mcp-crudo.py`** | diagnóstico | **AMPL + Gurobi** | reales | Idéntico al principal **salvo que NO re-balancea el MCP**: lee el `nsalas` tal cual del `.dat`. Sirve para el **diagnóstico del Capítulo 4** (déficit estructural, instancias infactibles, overtime forzado). Salida en `Databases/optimized-sets-real-crudo/` y `logs-real-crudo/`. |
| **`model-solver-real-debug.py`** | depuración | **AMPL + Gurobi** | reales | Resuelve **solo las instancias de control** (por defecto 11/17/18) volcando **toda la traza de Gurobi** (incluida la descomposición diaria de P2) al log. Para estudiar el comportamiento instancia a instancia. |
| **`model-solver-toy.py`** | verificación | AMPL + Gurobi | inventado | **Toy mínimo** (1 día, 7 pacientes) con óptimo calculable a mano. Verifica las restricciones (1)–(10) y revela el **acoplamiento visita→infusión** ($F_1=2$, no 1). |
| **`model-solver-toy-peor.py`** | verificación | AMPL + Gurobi | inventado | **Toy "peor caso"** (3 días, déficit estructural, recursos saturados) que reproduce a escala diminuta lo que hace difícil a P1. Valida la cota **LB1 = 3** y $F_1 = 6$. |
| **`generador_figuras.py`** | — | **matplotlib** | todos | Clase `GeneradorFiguras`: produce las **41 figuras** (PDF + PNG) del TFG a partir de resúmenes, agendas, `.dat` y logs, con los datos del artículo embebidos. `python generador_figuras.py` → `figuras/`. |

---

## Carpetas de datos, resultados y logs

| Carpeta | Contenido |
|---|---|
| `Databases/real-sets/` | **52 ficheros `.dat`** reales (`istanza01.dat`…`istanza52.dat`): pacientes (patología `alpha`, duración de visita `v`, de infusión `f`, criticidad `lambda`), MCP `w[r,k,d]`, `giornilav` y `c[i,j]` (festivos). |
| `Databases/simulated-sets/` | **30 `.csv`** sintéticos (`datos_simulados_v1.csv`…`v30`). |
| `Databases/optimized-sets/` | Resultados sintéticos: 90 `agenda_*.csv` + `resumen_ejecucion_ampl.csv`. |
| `Databases/optimized-sets-real/` | Resultados Opción C: 51 `agenda_istanzaNN.csv` + `resumen_ejecucion_real.csv`. |
| `Databases/optimized-sets-real-crudo/` | Resultados MCP crudo: 47 agendas (4 instancias infactibles no producen agenda) + `resumen_ejecucion_real_crudo.csv`. |
| `Databases/optimized-sets-real-debug/` | Resultados de las instancias de control (11/17/18) + `resumen_debug.csv`. |
| `logs-real/`, `logs-real-crudo/` | 51 logs de Gurobi cada uno (`run_istanzaNN.txt`): traza del solver, LB1, nodos B&B, gap, tiempos. |
| `logs-ampl/`, `logs/` | Logs de las ejecuciones simuladas (AMPL/Gurobi y PuLP/CBC). |

---

## Guía de reproducibilidad paso a paso

> Todos los comandos asumen el `venv` activo y ejecutarse **desde la raíz del repositorio**
> (salvo el generador de datos; ver nota).

```bash
# ── FASE 1: datos sintéticos ────────────────────────────────────────────────
# (0) Generar las 30 instancias sintéticas  [reproducible por semilla, §8]
cd Databases && python ../dataset-generator.py && cd ..
#    → escribe en ./simulated-sets/ (el script usa ruta relativa; ejecútalo desde Databases/)

# (1) Prototipo PuLP/CBC (opcional; ilustra el OOM en las fracciones grandes)
python model-solver.py

# ── FASE 2: validación en AMPL/Gurobi sobre datos sintéticos ────────────────
python model-solver-ampl.py
#    → Databases/optimized-sets/  +  logs-ampl/   (90 ejecuciones, F1=F2=0)

# ── FASE 3–7: datos reales ──────────────────────────────────────────────────
# (2) Barrido DEFINITIVO con la Opción C (MCP mensual re-balanceado)
python model-solver-real.py
#    → Databases/optimized-sets-real/  +  logs-real/   (51 instancias)

# (3) Barrido de DIAGNÓSTICO con el MCP crudo (sin re-balanceo)
python model-solver-real-mcp-crudo.py
#    → Databases/optimized-sets-real-crudo/  +  logs-real-crudo/

# (4) Depuración de las instancias de control (log completo)
python model-solver-real-debug.py

# ── VERIFICACIÓN (toys, óptimo a mano) ──────────────────────────────────────
python model-solver-toy.py          # 1 día, 7 pacientes  → F1=2
python model-solver-toy-peor.py     # 3 días, déficit     → LB1=3, F1=6

# ── FIGURAS ─────────────────────────────────────────────────────────────────
python generador_figuras.py
#    → figuras/   (41 figuras PDF + PNG)

# ── ANEXO DE PSEUDOCÓDIGO (LaTeX, opcional) ─────────────────────────────────
cd docs/pseudocodigo && latexmk -pdf anexo-pseudocodigo.tex && cd ../..
```

**Tiempos orientativos** (por instancia real): P1 ~ segundos (cierra en 1 nodo con
Opción C), P2 ~ segundos, P3 el más costoso; media ≈ 235 s/instancia, máximo ≈ 2 250 s en las
festivas. El barrido completo de 51 instancias son varias horas, en función del hardware disponible.

---

## La semilla y la generación de datos sintéticos

`dataset-generator.py` es **totalmente reproducible**:

```python
BASE_SEED = 20260427
for i in range(1, 31):                                  # 30 instancias (v1…v30)
    rng_local = np.random.default_rng(BASE_SEED + i)    # semilla por instancia
    num_pacientes = int(rng_local.integers(430, 720))   # tamaño ∈ [430, 719]
    generar_dataset_tfg(num_pacientes, version=i, seed=BASE_SEED + i)
```

Perfil de cada paciente (replica del hospital de referencia, según el artículo):

| Atributo | Distribución |
|---|---|
| Patología (cMG) | `HE 30 %`, `BR 26 %`, `LU 13 %`, `OT 12 %`, `GI 10 %`, `UR 6 %`, `GY 3 %` |
| Duración de visita `v_p` | **2 slots** (20 min) si HE; **1 slot** (10 min) en el resto |
| Duración de infusión `f_p` | uniforme discreta en **6–24 slots** (1–4 h) |
| Crítico (requiere cama) | `1` con probabilidad **0,2446**; si no, `0` |

Como cada instancia usa `seed = BASE_SEED + i`, **volver a ejecutar el script regenera exactamente
los mismos 30 ficheros**. (El generador escribe en `./simulated-sets/`; ejecútalo desde
`Databases/` o ajusta la ruta de salida.)

---

## Parámetros del modelo (constantes)

Definidos al inicio de `model-solver-real.py` (idénticos al artículo, §5.1):

| Constante | Valor | Significado |
|---|---|---|
| `NV` | 36 | último slot **regular** de visita (14:00) |
| `NI` | 54 | último slot **regular** de infusión (17:00) |
| `SLOTS_VISITA` | 1…48 | horizonte de visita ($|S^V|=48$: 36 + 12 de overtime) |
| `SLOTS_INFUSION` | 1…66 | horizonte de infusión ($|S^I|=66$: 54 + 12) |
| `MAX_SALAS` | 6 | salas de consulta del centro |
| `MAX_SILLONES` | 26 | sillones de infusión |
| `MAX_CAMAS` | 27 | camas de infusión |
| `TIME_LIMIT_P1/P2/P3` | 600 / 600 / 1200 s | límites de tiempo por subproblema |
| `TIME_LIMIT_DAY` | 120 s | límite del subproblema diario (Procedimiento 2) |
| `GAP_REL` | 0,05 | gap relativo de Gurobi (5 %) |
| `EXCLUDED_INSTANCES` | `{istanza14}` | única de las 52 que no figura en la Tabla 7 del artículo |

---

## Formato de los datos

**Entrada — `Databases/real-sets/istanzaNN.dat`** (formato AMPL):
- `param giornilav := 5;` — días laborables (5, o 4/3 en festivos).
- `param c : … ;` — `c[i,j]=1` si el `i`-ésimo día laborable es el `j`-ésimo día natural (festivos).
- `param w := [r,k,d] 1 …;` — MCP: sala `r` dedicada a la patología `k` el día `d`.
- `param: alpha v f lambda := …;` — por paciente: patología, visita, infusión, criticidad.

**Salida — `agenda_istanzaNN.csv`** (una fila por paciente):
`ID_Paciente, Critico, Dia, Slot_Inicio_Visita, Slot_Inicio_Infusion, Recurso_Infusion, Espera_Slots, Espera_Minutos`

**Resumen — `resumen_ejecucion_real*.csv`** (una fila por instancia):
`instancia, giornilav, n_pacientes, n_criticos, status_p1/p2/p3, f1_overtime_slots, f2_maxwait_slots, f3_sillones, pct_sillon, espera_media_min, t_p1_s, t_p2_s, t_p3_s, t_total_s, coherente, log, agenda`

---

## Notas de reproducibilidad y limitaciones

- **Dos máquinas / dos parches de Gurobi.** El banco simulado se ejecutó en **Windows 11 con
  Gurobi 13.0.1** junto con los barridos reales (Opción C y crudo). Los resultados agregados no dependen de esto, pero los tiempos y algún incumbente sí.
- **P1 no siempre cierra con el MCP crudo.** En 17/47 instancias del barrido crudo, P1 agota el
  tiempo: el `F1` reportado es un **incumbente** (cota superior), no el óptimo. La cantidad robusta
  es la cota inferior **LB1**. Por eso el mismo `istanza03` crudo dio F1=18 en un run y 24 en otro
  (misma cota LB1 = 15,3): es el efecto de terminar por límite de tiempo, no un error.
- **Licencia AMPL temporal.** El UUID incrustado es una licencia académica con caducidad; para
  reproducir hay que usar una licencia propia.
- **`istanza14` se excluye** del barrido (no figura en la Tabla 7 del artículo; huella
  `(612, 156)`).

---
