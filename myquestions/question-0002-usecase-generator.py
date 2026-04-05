import pandas as pd
import numpy as np

def generar_caso_de_uso_calcular_racha_maxima():
    np.random.seed()

    usuarios = ['U1', 'U2', 'U3']
    data = []

    for user in usuarios:
        n = np.random.randint(10, 20)

        fechas = pd.date_range(start="2023-01-01", periods=n, freq="D")
        np.random.shuffle(fechas)

        eventos = np.random.choice([0, 1], size=n, p=[0.6, 0.4])

        for f, e in zip(fechas, eventos):
            data.append({
                "usuario": user,
                "fecha": f,
                "evento": int(e)
            })

    df = pd.DataFrame(data)

    # -------------------------------
    # FUNCIÓN SOLUCIÓN (esperada)
    # -------------------------------
    def calcular_racha_maxima(df):
        resultados = []

        for user in df["usuario"].unique():
            subset = df[df["usuario"] == user].sort_values("fecha")

            max_racha = 0
            racha_actual = 0

            for val in subset["evento"]:
                if val == 1:
                    racha_actual += 1
                    max_racha = max(max_racha, racha_actual)
                else:
                    racha_actual = 0

            resultados.append({
                "usuario": user,
                "racha_maxima": int(max_racha)
            })

        return pd.DataFrame(resultados)

    # Ejecutar solución esperada
    output = calcular_racha_maxima(df)

    return {
        "input": df,
        "output_esperado": output
    }
