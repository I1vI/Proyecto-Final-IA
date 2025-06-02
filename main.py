import numpy as np

from src.carga_datos import cargar_datos
from src.preprocesamiento_de_datos import media_datos,label_encoder,normalizar_datos,balanceo_duplicando,balanceo_eliminando,cuenta_clases


#========= CARGAMOS EL DATASET =========
datos = cargar_datos()
x = datos.drop(columns=['Target'])
y = datos['Target']

#========= Proceso Básico de Análisis de Datos =========

#=======================================
# a) PREPROCESAMIENTO DE LOS DATOS

#++++++++++++++++++++++++++++++++++++++++
# INPUTAMOS LOS DATOS (EN X) CON LA MEDIA
x_media = media_datos(x)

#++++++++++++++++++++++++++++++++++++++++
# USAMOS LABEL ENCODER PARA EL CASO DEL TARGET, HAY 3 TIPOS (dropout, enrolled, graduate)
y_label_encoder, clases = label_encoder(y)
clases = clases.classes_

#++++++++++++++++++++++++++++++++++++++++
# NORMALIZAMOS LOS DATOS (EN X)
x_normalizado= normalizar_datos(x_media)

#++++++++++++++++++++++++++++++++++++++++
# BALANCEO DE DATOS (2 tipos)
x_balanceado, y_balanceado = balanceo_duplicando(x_normalizado, y_label_encoder)
#x_balanceado, y_balanceado = balanceo_eliminando(x_normalizado, y_label_encoder)

#cuenta_clases(y_balanceado,clases)

print("✅ Se termino con el preprocesamiento.")