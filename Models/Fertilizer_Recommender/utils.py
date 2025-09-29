import os
import joblib
import json

BASE_DIR =os.path.dirname(__file__)
MODEL_DIR=os.path.join(BASE_DIR , "models")

def save_model(obj, filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(obj, os.path.join(MODEL_DIR, filename))

def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None

def save_json(data, filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, filename), "w") as f:
        json.dump(data, f)

def load_json(filename):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None
