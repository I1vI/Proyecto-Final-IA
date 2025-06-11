import numpy as np
from src._1_carga_datos import cargar_datos
from src._2_A_preprocesamiento_de_datos import (
    media_datos,
    media_datos_manual,
    label_encoder,
    label_encoder_manual,
    normalizar_datos,
    balanceo_duplicando,
    balanceo_eliminando,
    cuenta_clases,
    estandarizar_datos
)
from src._2_C_ejecucion_del_modelo import (
    particionar_datos,
    entrenar_modelo,
    evaluar_modelo
)
from src._2_D_Splits_Validación_por_Asignaciones import(
    particionar_datos_split,
    entrenar_modelo_split,
    evaluar_modelo_split
)
from src._3_A_B_pca import(
    aplicar_pca, 
    explicar_varianza
)
from src._4_aprendizaje_no_supervisado import clustering


#================================================================
#================================================================
#                 1   CARGAMOS EL DATASET
#================================================================
#================================================================
datos = cargar_datos()
x = datos.drop(columns=['Target'])
y = datos['Target']

#================================================================
#================================================================
#            2  Proceso Básico de Análisis de Datos
#================================================================
#================================================================

#================================================================
#            a) PREPROCESAMIENTO DE LOS DATOS
#================================================================

#++++++++++++++++++++++++++++++++++++++++
# INPUTAMOS LOS DATOS (EN X) CON LA MEDIA
x_media = media_datos(x)
#x_media = media_datos_manual(x)


#++++++++++++++++++++++++++++++++++++++++
# USAMOS LABEL ENCODER PARA EL CASO DEL TARGET, HAY 3 TIPOS (dropout, enrolled, graduate)
y_label_encoder, clases = label_encoder(y)
clases = clases.classes_
#y_label_encoder, clases = label_encoder_manual(y)


#++++++++++++++++++++++++++++++++++++++++
# NORMALIZAMOS LOS DATOS (EN X)
x_normalizado = normalizar_datos(x_media)
#x_normalizado = estandarizar_datos(x_media)


#++++++++++++++++++++++++++++++++++++++++
# BALANCEO DE DATOS (2 tipos)
#cuenta_clases(y_label_encoder,clases)
x_balanceado, y_balanceado = balanceo_duplicando(x_normalizado, y_label_encoder)
#x_balanceado, y_balanceado = balanceo_eliminando(x_normalizado, y_label_encoder)
#cuenta_clases(y_balanceado,clases)

print("Se termino con el preprocesamiento.")



#================================================================
#            c) PRIMERA EJECUCION DEL MODELO
#================================================================

#++++++++++++++++++++++++++++++++++++++++
# CON EL ALGORITMO Random Forest Classifier HACEMOS UNA PRIMERA EVALUACION
x_train, x_test, y_train, y_test = particionar_datos(x_balanceado, y_balanceado,porcentaje=0.8)
modelo = entrenar_modelo(x_train, y_train)
evaluar_modelo(modelo, x_test, y_test)

print("Se termino la primera Ejecución del Modelo.")

#================================================================
#            d) VALIDACION POR ASIGNACIONES (SPLITS)
#================================================================
accuracies = []
precisions = []
recalls = []
f1_scores = []

#++++++++++++++++++++++++++++++++++++++++
# ACADEMICO: 80% entrenamiento y 20% prueba

for i in range(2):
    x_train, x_test, y_train, y_test = particionar_datos_split(x_balanceado, y_balanceado, porcentaje=0.8)
    modelo = entrenar_modelo_split(x_train, y_train)
    acc, prec, rec, f1 = evaluar_modelo_split(modelo, x_test, y_test)
    #print(f"Iteración {i+1:3}: Accuracy = {acc:.4f} | Precision = {prec:.4f} | Recall = {rec:.4f} | F1 = {f1:.4f}")
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

print("\nValidación por Asignaciones (80/20) completada con 100 splits.")
print(f"Mediana Accuracy : {np.median(accuracies):.4f}")
print(f"Mediana Precision: {np.median(precisions):.4f}")
print(f"Mediana Recall   : {np.median(recalls):.4f}")
print(f"Mediana F1-score : {np.median(f1_scores):.4f}")

#++++++++++++++++++++++++++++++++++++++++
# INVESTIGACION: 50% entrenamiento y 50% prueba

for i in range(2):
    x_train, x_test, y_train, y_test = particionar_datos_split(x_balanceado, y_balanceado, porcentaje=0.5)
    modelo = entrenar_modelo_split(x_train, y_train)
    acc, prec, rec, f1 = evaluar_modelo_split(modelo, x_test, y_test)
    #print(f"Iteración {i+1:3}: Accuracy = {acc:.4f} | Precision = {prec:.4f} | Recall = {rec:.4f} | F1 = {f1:.4f}")
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

print("\nValidación por Asignaciones (50/50) completada con 100 splits.")
print(f"Mediana Accuracy : {np.median(accuracies):.4f}")
print(f"Mediana Precision: {np.median(precisions):.4f}")
print(f"Mediana Recall   : {np.median(recalls):.4f}")
print(f"Mediana F1-score : {np.median(f1_scores):.4f}")

print("Se termino la validación por asignaciones (Splits).")




#================================================================
#================================================================
#             3 Reducción de Dimensionalidad con PCA
#================================================================
#================================================================
x_normalizado = estandarizar_datos(x_media)
x_balanceado, y_balanceado = balanceo_duplicando(x_normalizado, y_label_encoder)

#================================================================
#            a) Análisis de Componentes Principales (PCA)
#================================================================
print("\n=== Análisis de Componentes Principales (PCA) con 10 componentes===")
x_pca, pca_obj = aplicar_pca(x_balanceado, 10)
var_exp, var_exp_acum = explicar_varianza(pca_obj)
print(x_pca)
print("Varianza explicada por componente:", var_exp)
print("Varianza explicada acumulada:", var_exp_acum)


#================================================================
# b) componentes: (12, 10, 11, 9, 5, 3) y determinar la cantidad óptima 
#================================================================
componentes_a_probar = [14, 12, 10, 11, 9, 5, 3]
resultados = []
print("\n=== Evaluación con distintas cantidades de componentes PCA ===")
for n_comp in componentes_a_probar:
    print(f"\nProbando con {n_comp} componentes principales:")
    # Aplicar PCA
    x_pca, pca_obj = aplicar_pca(x_balanceado, n_comp)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    for i in range(50):
        x_train, x_test, y_train, y_test = particionar_datos_split(x_pca, y_balanceado, porcentaje=0.8)
        modelo = entrenar_modelo_split(x_train, y_train)
        acc, prec, rec, f1 = evaluar_modelo_split(modelo, x_test, y_test)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

    resultados.append({
        'componentes': n_comp,
        'accuracy': np.median(accuracies),
        'precision': np.median(precisions),
        'recall': np.median(recalls),
        'f1': np.median(f1_scores)
    })
    print(f"Mediana Accuracy : {np.median(accuracies):.4f}")
    print(f"Mediana Precision: {np.median(precisions):.4f}")
    print(f"Mediana Recall   : {np.median(recalls):.4f}")
    print(f"Mediana F1-score : {np.median(f1_scores):.4f}")


mejor_resultado = max(resultados, key=lambda x: x['f1'])
print("\n=== Mejor configuración encontrada ===")
print(f"Componentes principales: {mejor_resultado['componentes']}")
print(f"Mediana Accuracy : {mejor_resultado['accuracy']:.4f}")
print(f"Mediana Precision: {mejor_resultado['precision']:.4f}")
print(f"Mediana Recall   : {mejor_resultado['recall']:.4f}")
print(f"Mediana F1-score : {mejor_resultado['f1']:.4f}")



#================================================================
#================================================================
#             4 Aprendizaje No Supervisado
#================================================================
#================================================================



#================================================================
#        Usamos la Reducción de Dimensionalidad con PCA 
#================================================================

print(f"\n=== Análisis de Componentes Principales (PCA) con {mejor_resultado['componentes']} componentes ===")
x_pca, pca_obj = aplicar_pca(x_balanceado, mejor_resultado['componentes'])
var_exp, var_exp_acum = explicar_varianza(pca_obj)
print("Varianza explicada por componente:", var_exp)
print("Varianza explicada acumulada:", var_exp_acum)

#================================================================
#  Aplicar clustering no supervisado sobre x_balanceado (sin usar y_label_encoder)
#================================================================

df_clusters, scores_clusters = clustering(
    x_pca,
    y=y_balanceado,
    max_clusters=6,
    plot=True
)
