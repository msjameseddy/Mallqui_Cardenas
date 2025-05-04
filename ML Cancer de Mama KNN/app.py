from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Cargar modelo y columnas
modelo = pickle.load(open("modelo_cancer.pkl", "rb"))
columnas = pickle.load(open("columnas_modelo.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        datos = [float(x) for x in request.form.values()]
        entrada = pd.DataFrame([datos], columns=columnas)

        prediccion = modelo.predict(entrada)[0]
        probabilidad = modelo.predict_proba(entrada)[0]

        resultado = "Positivo para cáncer de mama" if prediccion == 1 else "Negativo para cáncer de mama"
        return render_template('index.html', prediction_text=resultado, prob=round(max(probabilidad)*100, 2))

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)