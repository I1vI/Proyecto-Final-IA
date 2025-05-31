import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def cargar_datos(ruta, separador=';'):
    """
    Carga un archivo CSV y lo devuelve como DataFrame.
    """
    try:
        df = pd.read_csv(ruta, sep=separador)
        print("✅ Datos cargados correctamente.")
        return df
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado en la ruta: {ruta}")
        return None

def mostrar_info_basica(df):
    """
    Muestra las primeras filas y la info general del DataFrame.
    """
    print("\n📋 Primeras filas del dataset:")
    print(df.head())
    print("\n📊 Información general del dataset:")
    print(df.info())

def imputar_valores_nulos(df):
    """
    Aplica imputación con la media a columnas numéricas del DataFrame.
    """
    datos_numericos = df.select_dtypes(include=[np.number])
    print("\n🔢 Matriz original con posibles valores nulos:")
    print(datos_numericos.values)

    imputador = SimpleImputer(strategy='mean')
    datos_imputados = imputador.fit_transform(datos_numericos)

    print("\n✅ Matriz después de imputación con media:")
    print(datos_imputados)

    # Reemplaza los datos imputados en el dataframe original
    df[datos_numericos.columns] = datos_imputados
    return df
