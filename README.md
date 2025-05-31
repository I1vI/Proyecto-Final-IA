<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png" alt="Python Logo" width="300"/>
</p>

# 🎓 Predicción de Deserción Estudiantil en Educación Superior

Este proyecto tiene como objetivo aplicar técnicas de aprendizaje automático para predecir el estado académico final de estudiantes universitarios (abandono, continuidad o graduación), utilizando datos académicos, personales y socioeconómicos recopilados desde el momento de su matrícula.

---

## 📊 Dataset Utilizado

El dataset empleado se titula **"Predict Students’ Dropout and Academic Success"** y proviene del [UCI Machine Learning Repository]. Fue creado en el marco de un proyecto en Portugal para reducir la deserción académica en la educación superior. Incluye información sobre:

- Calificaciones académicas  
- Datos personales (edad, género, nacionalidad)  
- Condiciones económicas y familiares (becas, situación laboral de los padres)  
- Indicadores macroeconómicos del país  

La variable objetivo es `Target`, con tres clases:
- 🎓 `Graduate` – El estudiante se graduó exitosamente  
- 📝 `Enrolled` – El estudiante continúa inscrito  
- ❌ `Dropout` – El estudiante abandonó  

<p align="center">
  <a href="https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success">
    <img src="https://img.shields.io/badge/Descargar%20Dataset-UCI%20Repository-blue?style=for-the-badge&logo=data" alt="Dataset UCI">
  </a>
</p>

---

## 🎯 Objetivo de Investigación

El objetivo de esta investigación es **detectar tempranamente a los estudiantes en riesgo de abandonar sus estudios**, mediante la construcción de modelos predictivos que analicen características académicas, personales y socioeconómicas.  
Con esta información se busca apoyar la toma de decisiones institucionales y reducir los índices de deserción en la educación superior.

---

## 🛠️ Tecnologías Utilizadas

- Python 3.x  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- VS Code  
- GitHub  

---

## 📁 Estructura del Proyecto

```plaintext
proyecto-desercion/
│
├── data/                  # Dataset original y preprocesado
│   └── estudiantes.csv
│
├── src/                   # Módulos de código (preprocesamiento, modelos, etc.)
│   └── preprocesamiento.py
│
├── resultados/            # Gráficos, métricas, CSVs generados
│   └── matriz_confusion.png
│
├── docs/                  # Artículos académicos (PDFs)
│   └── articulo_1.pdf
│
├── main.py                # Script principal del proyecto
├── README.md              # Este archivo
└── requirements.txt       # Librerías necesarias

```
## 🚀 Instrucciones para ejecutar el proyecto

```bash
# 1. Clonar el repositorio
git clone https://github.com/I1vI/Proyecto-Final-IA.git
cd Proyecto-Final-IA

# 2. Crear y activar entorno virtual
# En Windows:
python -m venv venv
venv\Scripts\activate

# En macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar el proyecto
python main.py

```

## 🚀 Instrucciones para ejecutar el proyecto

```bash
# 1. Clonar el repositorio
git clone https://github.com/I1vI/Proyecto-Final-IA.git

# 2. Crear y activar entorno virtual

# En Windows:
python -m venv venv
venv\Scripts\activate

# En macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar el proyecto
python main.py
