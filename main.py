import numpy as np

from src.carga_datos import cargar_datos
from src.preprocesamiento_de_datos import (
    media_datos,
    label_encoder,
    normalizar_datos,
    balanceo_duplicando,
    balanceo_eliminando,
    cuenta_clases
)
from src.ejecucion_del_modelo import (
    particionar_datos,
    entrenar_modelo,
    evaluar_modelo
)
from src.Validación_por_Asignaciones_Splits import(
    particionar_datos_split,
    entrenar_modelo_split,
    evaluar_modelo_split
)
from src.pca import(
    aplicar_pca, 
    explicar_varianza
)


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


#=======================================
# c) PRIMERA EJECUCION DEL MODELO

#++++++++++++++++++++++++++++++++++++++++
# CON EL ALGORITMO Random Forest Classifier HACEMOS UNA PRIMERA EVALUACION
x_train, x_test, y_train, y_test = particionar_datos(x_balanceado, y_balanceado,porcentaje=0.8)
modelo = entrenar_modelo(x_train, y_train)
evaluar_modelo(modelo, x_test, y_test)

print("✅ Se termino la primera Ejecución del Modelo.")

#=======================================
# d) VALIDACION POR ASIGNACIONES (SPLITS)
accuracies = []
precisions = []
recalls = []
f1_scores = []

#++++++++++++++++++++++++++++++++++++++++
# FINES ACADEMICOS: 80% entrenamiento y 20% prueba

for i in range(2):
    x_train, x_test, y_train, y_test = particionar_datos_split(x_balanceado, y_balanceado, porcentaje=0.8)
    modelo = entrenar_modelo_split(x_train, y_train)
    acc, prec, rec, f1 = evaluar_modelo_split(modelo, x_test, y_test)
    #print(f"Iteración {i+1:3}: Accuracy = {acc:.4f} | Precision = {prec:.4f} | Recall = {rec:.4f} | F1 = {f1:.4f}")
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

print("\n✅ Validación por Asignaciones (80/20) completada con 100 splits.")
print(f"Mediana Accuracy : {np.median(accuracies):.4f}")
print(f"Mediana Precision: {np.median(precisions):.4f}")
print(f"Mediana Recall   : {np.median(recalls):.4f}")
print(f"Mediana F1-score : {np.median(f1_scores):.4f}")

#++++++++++++++++++++++++++++++++++++++++
# FINES ACADEMICOS: 50% entrenamiento y 50% prueba

for i in range(2):
    x_train, x_test, y_train, y_test = particionar_datos_split(x_balanceado, y_balanceado, porcentaje=0.5)
    modelo = entrenar_modelo_split(x_train, y_train)
    acc, prec, rec, f1 = evaluar_modelo_split(modelo, x_test, y_test)
    #print(f"Iteración {i+1:3}: Accuracy = {acc:.4f} | Precision = {prec:.4f} | Recall = {rec:.4f} | F1 = {f1:.4f}")
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

print("\n✅ Validación por Asignaciones (50/50) completada con 100 splits.")
print(f"Mediana Accuracy : {np.median(accuracies):.4f}")
print(f"Mediana Precision: {np.median(precisions):.4f}")
print(f"Mediana Recall   : {np.median(recalls):.4f}")
print(f"Mediana F1-score : {np.median(f1_scores):.4f}")

print("✅ Se termino la validación por asignaciones (Splits).")



#========= Reducción de Dimensionalidad con PCA =========
#=======================================
# a) Análisis de Componentes Principales (PCA)

print("\n=== Análisis de Componentes Principales (PCA) con 10 componentes===")
x_pca, pca_obj = aplicar_pca(x_balanceado, 10)
var_exp, var_exp_acum = explicar_varianza(pca_obj)
print(x_pca)
print("Varianza explicada por componente:", var_exp)
print("Varianza explicada acumulada:", var_exp_acum)



#=======================================
# b) componentes: (12, 10, 11, 9, 5, 3) y determinar la cantidad óptima 
componentes_a_probar = [14, 12, 10, 11, 9, 5, 3]

print("\n=== Evaluación con distintas cantidades de componentes PCA ===")
for n_comp in componentes_a_probar:
    print(f"\nProbando con {n_comp} componentes principales:")
    # Aplicar PCA
    x_pca, pca_obj = aplicar_pca(x_balanceado, n_comp)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    for i in range(5):
        x_train, x_test, y_train, y_test = particionar_datos_split(x_pca, y_balanceado, porcentaje=0.8)
        modelo = entrenar_modelo_split(x_train, y_train)
        acc, prec, rec, f1 = evaluar_modelo_split(modelo, x_test, y_test)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

    print(f"Mediana Accuracy : {np.median(accuracies):.4f}")
    print(f"Mediana Precision: {np.median(precisions):.4f}")
    print(f"Mediana Recall   : {np.median(recalls):.4f}")
    print(f"Mediana F1-score : {np.median(f1_scores):.4f}")
