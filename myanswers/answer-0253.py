from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np

def encontrar_umbral_optimo(X, y, n_umbrales=50, random_state=42):

    X_scaled = StandardScaler().fit_transform(X)

    modelo = LogisticRegression(max_iter=1000, random_state=random_state)
    modelo.fit(X_scaled, y)

    probs = modelo.predict_proba(X_scaled)[:, 1]

    umbrales = np.linspace(0.01, 0.99, n_umbrales)

    mejor_f1 = -1
    mejor_umbral = umbrales[0]

    for u in umbrales:
        y_pred = (probs >= u).astype(int)
        f1 = f1_score(y, y_pred, zero_division=0)
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = u

    return round(float(mejor_umbral), 4)
