from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

def particionar_datos_split(x, y, porcentaje):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    Retorna:
    - x_train, x_test, y_train, y_test
    """
    return train_test_split(x, y, train_size=porcentaje)

def entrenar_modelo_split(x_train, y_train):
    """
    Entrena el modelo Random Forest con los datos de entrenamiento.
    """
    modelo = RandomForestClassifier()
    modelo.fit(x_train, y_train)
    return modelo

def evaluar_modelo_split(modelo, x_test, y_test):
    """
    Evalúa el modelo y retorna las métricas: accuracy, precision, recall, f1.
    """
    y_pred = modelo.predict(x_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    return accuracy, precision, recall, f1
