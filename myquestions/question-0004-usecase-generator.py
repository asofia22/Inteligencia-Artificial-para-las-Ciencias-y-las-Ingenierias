import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def generar_caso_de_uso_entrenar_modelo_polinomial():
    np.random.seed(42)

    X = np.random.rand(100, 2)
    y = 3 * X[:, 0]**2 + 2 * X[:, 1] + np.random.normal(0, 0.1, 100)

    def entrenar_modelo_polinomial(X, y):
        modelo = Pipeline([
            ("poly", PolynomialFeatures(degree=2)),
            ("lr", LinearRegression())
        ])

        modelo.fit(X, y)
        preds = modelo.predict(X)

        return preds

    output = entrenar_modelo_polinomial(X, y)

    return (
        {
            "input": (X, y),
            "output": output
        },
        None
    )
