from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preparar_datos_clasificacion(df, columna_objetivo, test_size=0.25, random_state=42):

    df_c = df.dropna(subset=[columna_objetivo]).copy()

    X = df_c.drop(columns=[columna_objetivo])
    y = LabelEncoder().fit_transform(df_c[columna_objetivo])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
