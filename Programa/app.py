from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Ruta al modelo
model_path = os.path.join('model', 'rf_model_vivienda.pkl')
model = joblib.load(model_path)

model_path_comercio = os.path.join('model', 'modelo_svm_comercio.pkl')
model_comercio = joblib.load(model_path_comercio)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    dias = np.array([[data['dia']]])  # Aceptar un solo valor de 'día'
    prediction = model.predict(dias)

    prediction_comercio = model_comercio.predict(dias)

    response = {
            'Para el día': int(dias),
            'Generación de residuos vivienda': float(prediction),
            'Generación de residuos comercios': float(prediction_comercio)
        }

    # return jsonify({'generacion_residuos': prediction[0]})

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)