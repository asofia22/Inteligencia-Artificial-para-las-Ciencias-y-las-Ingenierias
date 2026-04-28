from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge

def predecir_desgaste(X, y, umbral_varianza=0.01):

    n_features_originales = X.shape[1]

    vt = VarianceThreshold(threshold=umbral_varianza)
    X_sel = vt.fit_transform(X)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_sel)

    model = BayesianRidge()
    model.fit(X_scaled, y)

    r2 = round(model.score(X_scaled, y), 4)

    return {
        "n_features_originales": n_features_originales,
        "n_features_seleccionadas": X_sel.shape[1],
        "r2_train": r2,
        "coeficientes": model.coef_
    }
