from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Carga del modelo y preprocesador
with open('modelo_svm.pkl', 'rb') as f:
    modelo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('columnas.pkl', 'rb') as f:
    columnas = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtiene los valores del formulario
        input_data = [float(request.form[col]) for col in columnas]
        
        # Convierte en DataFrame
        datos = pd.DataFrame([input_data], columns=columnas)
        
        # Escala los datos
        datos_escalados = scaler.transform(datos)

        # Predicción
        pred = modelo.predict(datos_escalados)[0]
        proba = modelo.predict_proba(datos_escalados)[0]

        resultado = "Maligno" if pred == 1 else "Benigno"
        confianza = round(proba[pred] * 100, 2)

        return render_template('index.html', prediction_text=f'{resultado} con {confianza}% de confianza')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
