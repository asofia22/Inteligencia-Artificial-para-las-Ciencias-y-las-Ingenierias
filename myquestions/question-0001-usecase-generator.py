import pandas as pd
import numpy as np

def generar_caso_de_uso_detectar_outliers_por_grupo():
    np.random.seed(42)

    categorias = ['A', 'B', 'C']
    data = []

    for cat in categorias:
        valores = np.random.normal(loc=50, scale=10, size=20)

        valores = list(valores)
        valores.append(200)
        valores.append(-50)

        for v in valores:
            data.append({
                "categoria": cat,
                "valor": float(v)
            })

    df = pd.DataFrame(data)

    def detectar_outliers_por_grupo(df):
        df_result = df.copy()
        df_result["es_outlier"] = False

        for cat in df["categoria"].unique():
            subset = df[df["categoria"] == cat]

            Q1 = subset["valor"].quantile(0.25)
            Q3 = subset["valor"].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            mask = (
                (df_result["categoria"] == cat) &
                (
                    (df_result["valor"] < lower) |
                    (df_result["valor"] > upper)
                )
            )

            df_result.loc[mask, "es_outlier"] = True

        return df_result

    output = detectar_outliers_por_grupo(df)

    return (
        {
            "input": df,
            "output": output
        },
    )
