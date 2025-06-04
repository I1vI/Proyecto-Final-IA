from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

def particionar_datos(x, y,porcentaje):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    Retorna:
    - X_train, X_test, y_train, y_test
    """
    return train_test_split(x, y, train_size=porcentaje, random_state=42)


def entrenar_modelo(x_train, y_train):
    """
    Entrenamos el modelo con x_train, y_train
    """
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(x_train, y_train)
    return modelo


def evaluar_modelo(modelo, X_test, y_test):
    """
    Evalúa el modelo con el conjunto de prueba.
    """

    y_pred = modelo.predict(X_test)

    # Accuracy = (número de predicciones correctas) / (número total de ejemplos)
    accuracy = accuracy_score(y_test, y_pred)
    # Precision = promedio de [TP / (TP + FP)]
    precision = precision_score(y_test, y_pred, average='macro')
    # Recall = promedio de [TP / (TP + FN)]
    recall = recall_score(y_test, y_pred, average='macro')
    # F1-score = promedio de [2 * (Precision * Recall) / (Precision + Recall)]
    f1 = f1_score(y_test, y_pred, average='macro')

    print("====== MÉTRICAS DEL MODELO ======")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # Reporte por clase
    print("\n====== REPORTE DE CLASIFICACIÓN ======")
    print(classification_report(y_test, y_pred, target_names=["Dropout", "Enrolled", "Graduate"]))

    # Matriz de confusión
    print("====== MATRIZ DE CONFUSIÓN ======")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
