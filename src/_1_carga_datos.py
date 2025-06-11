
import pandas as pd

def cargar_datos():

    ruta = 'data/data.csv'
    separador=';'
    
    try:
        df = pd.read_csv(ruta, sep=separador)
        print("Datos cargados correctamente.")
        return df
    except FileNotFoundError:
        print(f"Archivo no encontrado en la ruta: {ruta}")
        return None
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None
