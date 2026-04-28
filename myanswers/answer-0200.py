from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def entrenar_modelo_precio_vivienda(df, target_col):

    # 1. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    # 2. Train / Test split (igual que el generador)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Entrenar modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # 4. Predicciones
    y_pred = modelo.predict(X_test)

    # 5. MSE
    mse = mean_squared_error(y_test, y_pred)

    return modelo, mse
