import pandas as pd
import numpy as np


def generar_dataset_tfg(num_pacientes=614, version=1, seed=12345):
    rng = np.random.default_rng(seed)

    patologias = ["HE", "BR", "OT", "LU", "GI", "UR", "GY"]
    probabilidades = [0.30, 0.26, 0.12, 0.13, 0.10, 0.06, 0.03]

    data = []
    for i in range(num_pacientes):
        cmg = rng.choice(patologias, p=probabilidades)
        v_p = 2 if cmg == "HE" else 1
        f_p = rng.integers(6, 25)
        es_critico = 1 if rng.random() < 0.2446 else 0

        data.append(
            {
                "ID_Paciente": f"P{i+1:03}",
                "Patologia": cmg,
                "Visita_Slots": v_p,
                "Infusion_Slots": f_p,
                "Critico_Cama": es_critico,
            }
        )

    df = pd.DataFrame(data)
    nombre_archivo = f"datos_simulados_v{version}.csv"
    df.to_csv(f"./simulated-sets/{nombre_archivo}", index=False)
    print(f"Archivo {nombre_archivo} guardado con exito.")
    return df


BASE_SEED = 20260427

for i in range(1, 31):
    rng_local = np.random.default_rng(BASE_SEED + i)
    num_pacientes = int(rng_local.integers(430, 720))
    generar_dataset_tfg(num_pacientes=num_pacientes, version=i, seed=BASE_SEED + i)

