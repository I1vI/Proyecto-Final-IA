from src.preprocesamiento import cargar_datos, mostrar_info_basica, imputar_valores_nulos

# Paso 1: Cargar dataset
ruta = 'data/data.csv'
df = cargar_datos(ruta)

if df is not None:
    # Paso 2: Mostrar información inicial
    mostrar_info_basica(df)

    # Paso 3: Imputar valores nulos
    df = imputar_valores_nulos(df)
