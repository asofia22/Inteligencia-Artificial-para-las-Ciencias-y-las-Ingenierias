import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def generar_caso_de_uso_detectar_anomalias():
    np.random.seed(42)

    # Generar datos normales
    normales = np.random.normal(loc=0, scale=1, size=(100, 3))

    # Generar anomalías artificiales
    anomalias = np.random.uniform(low=8, high=10, size=(10, 3))

    # Combinar datos
    datos = np.vstack([normales, anomalias])

    df = pd.DataFrame(datos, columns=["f1", "f2", "f3"])

    # -------------------------------
    # FUNCIÓN SOLUCIÓN (esperada)
    # -------------------------------
    def detectar_anomalias(df):
        modelo = IsolationForest()
        modelo.fit(df)
        preds = modelo.predict(df)
        return preds

    # Ejecutar solución esperada
    output = detectar_anomalias(df)

    return {
        "input": df,
        "output_esperado": output
    }
