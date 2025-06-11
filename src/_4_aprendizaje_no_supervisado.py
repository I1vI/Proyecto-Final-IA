import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def clustering(x_pca, y=None, max_clusters=6, plot=True):
    scores = []
    for k in range(2, max_clusters + 1):
        modelo = KMeans(n_clusters=k, random_state=42)
        labels = modelo.fit_predict(x_pca)
        score = silhouette_score(x_pca, labels)
        scores.append((k, score))

    mejor_k = max(scores, key=lambda x: x[1])[0]
    
    print(f"Mejor número de clusters encontrado: k = {mejor_k}")

    final_model = KMeans(n_clusters=mejor_k, random_state=42)
    cluster_labels = final_model.fit_predict(x_pca)

    df_resultado = pd.DataFrame(x_pca, columns=[f'PC{i+1}' for i in range(x_pca.shape[1])])
    df_resultado['Cluster'] = cluster_labels

    if y is not None:
        df_resultado['Target'] = y.values

    if plot and x_pca.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_resultado, x='PC1', y='PC2', hue='Cluster', palette='Set2')
        plt.title(f'Clusters con PCA precalculado (k={mejor_k})')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True)
        plt.show()

        

    if 'Target' in df_resultado.columns:
        print("\nDistribución de clases reales por cluster (proporción por fila):")
        tabla = pd.crosstab(df_resultado['Cluster'], df_resultado['Target'], normalize='index')
        print(tabla.round(2))

    return df_resultado, scores

def clustering2(x_pca, y=None, max_clusters=6, plot=True):
     # Inicializamos una lista para almacenar los resultados del método del codo
    inertia_scores = []
    
    # Calcular la inertia (WCSS) para diferentes valores de K
    for k in range(2, max_clusters + 1):
        modelo = KMeans(n_clusters=k, random_state=42)
        modelo.fit(x_pca)
        inertia_scores.append(modelo.inertia_)
    
    # Graficar el método del codo
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), inertia_scores, marker='o', linestyle='--')
    plt.title('Método del Codo (Elbow Method)')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inertia (WCSS)')
    plt.grid(True)
    plt.show()
    
    # Elegir el número de clusters con el menor WCSS
    # Este método se basa en la observación visual del "codo"
    # Para la implementación automática, podrías usar una técnica de detección de codo, pero generalmente es visual
    # Una aproximación es buscar la reducción más pequeña en la pendiente de la curva.
    
    # Cálculo del Silhouette Score para determinar el mejor K
    scores = []
    for k in range(2, max_clusters + 1):
        modelo = KMeans(n_clusters=k, random_state=42)
        labels = modelo.fit_predict(x_pca)
        score = silhouette_score(x_pca, labels)
        scores.append((k, score))

    # Usar el K con el mayor Silhouette Score
    mejor_k = max(scores, key=lambda x: x[1])[0]
    
    print(f"Mejor número de clusters encontrado: k = {mejor_k}")
    
    # Modelo final con el mejor número de clusters
    final_model = KMeans(n_clusters=mejor_k, random_state=42)
    cluster_labels = final_model.fit_predict(x_pca)

    # Crear el DataFrame con los resultados
    df_resultado = pd.DataFrame(x_pca, columns=[f'PC{i+1}' for i in range(x_pca.shape[1])])
    df_resultado['Cluster'] = cluster_labels

    if y is not None:
        df_resultado['Target'] = y.values

    # Graficar los resultados si es posible visualizar en 2D
    if plot and x_pca.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_resultado, x='PC1', y='PC2', hue='Cluster', palette='Set2')
        plt.title(f'Clusters con PCA precalculado (k={mejor_k})')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True)
        plt.show()

    # Mostrar distribución de clases reales por cluster
    if 'Target' in df_resultado.columns:
        print("\nDistribución de clases reales por cluster (proporción por fila):")
        tabla = pd.crosstab(df_resultado['Cluster'], df_resultado['Target'], normalize='index')
        print(tabla.round(2))

    return df_resultado, scores