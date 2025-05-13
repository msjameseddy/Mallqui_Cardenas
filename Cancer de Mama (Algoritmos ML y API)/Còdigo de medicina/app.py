from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Cargar modelo y columnas
modelo = pickle.load(open("modelo_knn_acv.pkl", "rb"))
columnas = pickle.load(open("columnas_modelo.pkl", "rb"))

from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Cargar modelo y columnas correctos
modelo = pickle.load(open("modelo_knn_acv.pkl", "rb"))
columnas = pickle.load(open("columnas_modelo.pkl", "rb"))

# Umbrales de riesgo para interpretación
riesgos = [
    (0.4, 'BAJO', 'Seguimiento rutinario'),
    (0.7, 'MODERADO', 'Monitoreo intensivo'),
    (1.0, 'ALTO', 'Intervención inmediata')
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        datos = [float(x) for x in request.form.values()]
        entrada = pd.DataFrame([datos], columns=columnas)

        # Realizar predicción
        prediccion = modelo.predict(entrada)[0]
        probabilidad = modelo.predict_proba(entrada)[0][1]  # Probabilidad de clase "SI"

        # Evaluar el nivel de riesgo
        for umbral, nivel, accion in riesgos:
            if probabilidad <= umbral:
                resultado = f"Nivel de riesgo: {nivel} - Acción recomendada: {accion}"
                break

        return render_template(
            'index.html',
            prediction_text=resultado,
            prob=round(probabilidad * 100, 2)
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)