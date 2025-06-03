import pandas as pd
from sklearn.decomposition import PCA

def aplicar_pca(x, n_componentes):
    """
    Aplica PCA al dataset x con n_componentes y devuelve un DataFrame con los datos transformados
    manteniendo índices.
    """
    pca = PCA(n_components=n_componentes)
    x_pca_array = pca.fit_transform(x)
    columnas_pca = [f'PC{i+1}' for i in range(n_componentes)]
    x_pca_df = pd.DataFrame(x_pca_array, columns=columnas_pca, index=x.index)
    return x_pca_df, pca


def explicar_varianza(pca):
    """
    Retorna la varianza explicada acumulada por los componentes principales.
    """
    var_exp = pca.explained_variance_ratio_
    var_exp_acum = var_exp.cumsum()
    return var_exp, var_exp_acum
