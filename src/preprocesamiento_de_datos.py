import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def media_datos(X):
    """
    Imputa los datos con la media (En este caso imputa todas las X)
    """
    X_imputado = X.copy()
    imputer = SimpleImputer(strategy='mean')
    X_imputado[:] = imputer.fit_transform(X_imputado)
    return X_imputado


def label_encoder(y):
    """
    Dado que en el Target se tiene strings, usamos el label enconder para tener numeros, en este caso es 0,1,2
    """
    le = LabelEncoder()
    y_codificado_array = le.fit_transform(y)
    y_codificado = pd.Series(y_codificado_array, index=y.index, name=y.name)
    return y_codificado, le



def normalizar_datos(X):
    """
    Normalizamos los datos (las x) para manejarlo en un rango de 0 y 1
    """
    scaler = MinMaxScaler()
    X_normalizado_array = scaler.fit_transform(X)
    X_normalizado = pd.DataFrame(X_normalizado_array, columns=X.columns, index=X.index)
    return X_normalizado


def cuenta_clases(y,clases):
    """
    Verificamos la cantidad de clases que se tenga, para ver si necesita un balanceo o no
    """
    conteos = np.bincount(y)
    for i in range(len(conteos)):
        clase_original = clases[i]
        cantidad = conteos[i]
        print(f"{clase_original} (clase {i}): {cantidad} ejemplos")


def balanceo_duplicando(X, y):
    """
    Balanceamos duplicando los datos (La clasificacion minoritaria).
    """
    ros = RandomOverSampler(random_state=42)
    X_balanceado, y_balanceado = ros.fit_resample(X, y)
    return X_balanceado, y_balanceado


def balanceo_eliminando(X, y):
    """
    Balancea eliminando los datos (La clasificacion mayoritaria).
    """
    rus = RandomUnderSampler(random_state=42)
    X_balanceado, y_balanceado = rus.fit_resample(X, y)
    return X_balanceado, y_balanceado
