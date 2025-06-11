import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,StandardScaler
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

def media_datos_manual(X):
    """
    Imputa los datos con la media (En este caso imputa todas las X) manualmente
    """
    X_imputado = X.copy()
    
    for col in X.columns:
        suma = 0
        cantidad = 0
        
        for valor in X[col]:
            if pd.isna(valor):
                continue
            else:
                suma += valor
                cantidad += 1

        if cantidad != 0:
            media = suma / cantidad
        else:
            media = 0
        
        nueva_columna = []
        for valor in X[col]:
            if pd.isna(valor):
                nuevo_valor = media
            else:
                nuevo_valor = valor
            nueva_columna.append(nuevo_valor)
        X_imputado[col] = nueva_columna
    
    return X_imputado


def label_encoder(y):
    """
    Dado que en el Target se tiene strings, usamos el label enconder para tener numeros, en este caso es 0,1,2
    """
    le = LabelEncoder()
    y_codificado_array = le.fit_transform(y)
    y_codificado = pd.Series(y_codificado_array, index=y.index, name=y.name)
    return y_codificado, le

def label_encoder_manual(y):
    """
    Dado que en el Target se tiene strings, usamos el label enconder para tener numeros, en este caso es 0,1,2 manual
    """
    y_codificado = []
    mapa = {}
    codigo_actual = 0

    for valor in y:
        if valor not in mapa:
            mapa[valor] = codigo_actual
            codigo_actual += 1
        y_codificado.append(mapa[valor])

    y_codificado = pd.Series(y_codificado, index=y.index, name=y.name)
    return y_codificado, mapa


def normalizar_datos(X):
    """
    Normalizamos los datos (las x) para manejarlo en un rango de 0 y 1
    """
    scaler = MinMaxScaler()
    X_normalizado_array = scaler.fit_transform(X)
    X_normalizado = pd.DataFrame(X_normalizado_array, columns=X.columns, index=X.index)
    return X_normalizado

def estandarizar_datos(X):
    """
    Estandariza los datos (las X) para que tengan media 0 y desviación estándar 1,
    útil para algoritmos como clustering.
    """
    scaler = StandardScaler()
    X_estandarizado_array = scaler.fit_transform(X)
    X_estandarizado = pd.DataFrame(X_estandarizado_array, columns=X.columns, index=X.index)
    return X_estandarizado

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
