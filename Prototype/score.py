import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
    global model, scaler
    model_path = Model.get_model_path('/Users/fahmi.taib/Desktop/Deployment Code Test/Prototype/best_route_model.pkl')
    scaler_path = Model.get_model_path('/Users/fahmi.taib/Desktop/Deployment Code Test/Prototype/scaler.pkl')
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

def run(raw_data):
    data = json.loads(raw_data)['data']
    df = pd.DataFrame(data)
    X_scaled = scaler.transform(df)
    preds = model.predict(X_scaled)
    return json.dumps(preds.tolist())



