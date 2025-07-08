import joblib
import os
import pandas as pd
import json

def model_fn(model_dir):
    """ Carga el modelo desde el artefacto """
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)

def input_fn(input_data, content_type):
    """ Procesa la entrada en formato JSON """
    if content_type == "application/json":
        # Cargar JSON como dict
        data = json.loads(input_data)
        # Convertir a DataFrame
        return pd.DataFrame(data["instances"])
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """ Realiza predicciones usando el modelo cargado """
    return model.predict(input_data)

def output_fn(prediction, accept):
    """ Devuelve la predicción como JSON """
    if accept == "application/json":
        return json.dumps({"predictions": prediction.tolist()})
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
