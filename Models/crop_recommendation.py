import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from langchain_google_genai import ChatGoogleGenerativeAI

# ------------------ CONFIG ------------------
BASE_DIR = os.path.dirname(__file__)   # path to Models/
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, "ensemble_model.pkl")
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

DATA_PATH = os.path.join(BASE_DIR, "Crop_recommendation.csv")

# ------------------ LOAD DATA ------------------
data = pd.read_csv(DATA_PATH)
X = data.drop("label", axis=1)
y = data["label"]

# ------------------ TRAIN OR LOAD MODELS ------------------
def train_or_load_models():
    # Label encoder
    if os.path.exists(LE_PATH):
        le = joblib.load(LE_PATH)
        y_encoded = le.transform(y)
    else:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        joblib.dump(le, LE_PATH)

    # Random Forest
    if os.path.exists(RF_MODEL_PATH):
        rf_model = joblib.load(RF_MODEL_PATH)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
        rf_model.fit(X_train, y_train)
        joblib.dump(rf_model, RF_MODEL_PATH)

    # XGBoost
    if os.path.exists(XGB_MODEL_PATH):
        xgb_model = joblib.load(XGB_MODEL_PATH)
    else:
        X1_train, X1_test, y1_train, y1_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        xgb_model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss"
        )
        xgb_model.fit(X1_train, y1_train)
        joblib.dump(xgb_model, XGB_MODEL_PATH)

    # Ensemble
    if os.path.exists(ENSEMBLE_MODEL_PATH):
        ensemble_model = joblib.load(ENSEMBLE_MODEL_PATH)
    else:
        X2_train, X2_test, y2_train, y2_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        ensemble_model = VotingClassifier(
            estimators=[("rf", rf_model), ("xgb", xgb_model)],
            voting="soft"
        )
        ensemble_model.fit(X2_train, y2_train)
        joblib.dump(ensemble_model, ENSEMBLE_MODEL_PATH)

    return rf_model, xgb_model, ensemble_model, le

rf_model, xgb_model, ensemble_model, le = train_or_load_models()

# ------------------ PREDICTION FUNCTION ------------------
def predict_top3_crops_ensemble(model, features, label_encoder):
    probs = model.predict_proba([features])[0]
    top3_idx = np.argsort(probs)[-3:][::-1]
    results = []
    for idx in top3_idx:
        crop = label_encoder.inverse_transform([idx])[0]
        results.append((crop, round(probs[idx]*100, 2)))
    return results

# ------------------ LLM SETUP ------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    google_api_key="AIzaSyA5psEV1VtE2K468FmV5sCZyQ1sbUkIfgQ",
    temperature=0.3
)

crop_schema = {
  "title": "CropRecommendations",
  "type": "object",
  "properties": {
    "recommendations": {
      "type": "array",
      "minItems": 3,
      "maxItems": 3,
      "items": {
        "type": "object",
        "properties": {
          "crop": { "type": "string" },
          "confidence": { "type": "number" },
          "short_description": { "type": "string" },
          "detailed_description": {
            "type": "object",
            "properties": {
              "irrigation": { "type": "string" }
            },
            "required": ["irrigation"]
          }
        },
        "required": ["crop", "confidence", "short_description", "detailed_description"]
      }
    }
  },
  "required": ["recommendations"]
}

structured_llm = llm.with_structured_output(schema=crop_schema)

# ------------------ SINGLE FUNCTION ------------------
def predict_crop_llm(N, P, K, temperature, humidity, ph, rainfall):
    test_input = [N, P, K, temperature, humidity, ph, rainfall]
    predictions = predict_top3_crops_ensemble(ensemble_model, test_input, le)
    crops = [crop for crop, _ in predictions]
    percentages = [float(score) for _, score in predictions]

    predictions_dict = {
        "recommendations": [
            {"crop": crops[0], "confidence": percentages[0]},
            {"crop": crops[1], "confidence": percentages[1]},
            {"crop": crops[2], "confidence": percentages[2]}
        ]
    }

    prompt = f"""
    You are an agricultural expert.
    The ML model suggested these top crops with confidence scores:
    {predictions_dict}
    For each crop, generate:
    - A short description
    - Detailed description including soil/fertilizer, irrigation, pests/diseases, and market insights.
    Return ONLY valid JSON that follows the given schema.
    """
    response = structured_llm.invoke(prompt)
    return response
