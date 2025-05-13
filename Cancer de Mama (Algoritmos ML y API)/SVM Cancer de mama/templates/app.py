from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelo y columnas
modelo = joblib.load('modelo_knn_acv.pkl')
columnas = joblib.load('columnas_modelo.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        datos = [float(request.form[col]) for col in columnas]
        prob = modelo.predict_proba([datos])[0][1]
        
        # Evaluar el riesgo
        if prob <= 0.4:
            nivel = 'BAJO'
            accion = 'Seguimiento rutinario'
        elif prob <= 0.7:
            nivel = 'MODERADO'
            accion = 'Monitoreo intensivo'
        else:
            nivel = 'ALTO'
            accion = 'Intervención inmediata'

        resultado = f'Nivel de riesgo: {nivel} - Probabilidad: {prob*100:.2f}% - Acción: {accion}'
        return render_template('index.html', prediction_text=resultado)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)