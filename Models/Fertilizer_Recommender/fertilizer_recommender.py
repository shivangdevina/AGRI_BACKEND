import numpy as np
import pandas as pd
from Models.Fertilizer_Recommender.utils import load_model, load_json

# Auto-load saved models
ensemble_model = load_model("ensemble_model.pkl")
le = load_model("label_encoder.pkl")
X_columns = load_json("feature_columns.json")

if ensemble_model is None or le is None or X_columns is None:
    raise RuntimeError("❌ Models not found! Run train.py first.")

def recommend_fertilizers(N, P, K, temperature, humidity, ph, rainfall, crop, verbose=True):
    """
    Recommend top 3 fertilizers based on soil & weather conditions.
    User only passes input values + crop.
    """
    # Build input row
    example = pd.DataFrame([{
        "N": N, "P": P, "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
        "yield": 0
    }])

    # Handle crop one-hot encoding
    for col in X_columns:
        if col.startswith("label_"):
            example[col] = 0
    chosen_label = f"label_{crop}"
    if chosen_label in X_columns:
        example[chosen_label] = 1
    elif verbose:
        print(f"⚠️ Crop '{crop}' not found in training data.")

    # Ensure correct column order
    for col in X_columns:
        if col not in example.columns:
            example[col] = 0
    example = example[X_columns]

    # Predict probabilities
    probs = ensemble_model.predict_proba(example)
    top3_idx = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
    top3_probs = np.take_along_axis(probs, top3_idx, axis=1)[0]
    top3_names = le.inverse_transform(top3_idx.ravel()).reshape(top3_idx.shape)[0]

    results = [(str(name), float(p)) for name, p in zip(top3_names, top3_probs)]

    # Print friendly output
    if verbose:
        print(f"\n🌾 Fertilizer Recommender for {crop.title()} 🌾")
        
        for i, (fert, prob) in enumerate(results):
            print(f"{i+1:2d} {fert:<18} → {prob*100:.2f}%")

    # Return structured dict
    structured_output = {"recommendations": []}
    for fert, prob in results:
        structured_output["recommendations"].append({
            "fertilizer": fert,
            "confidence": round(prob*100, 2),
            "short_description": f"{fert} is recommended for {crop}.",
            "detailed_description": {
                "benefits": ["Improves yield", "Supplies nutrients"],
                "precautions": ["Follow dosage", "Avoid overuse"]
            }
        })

    return {"structured": structured_output}
# response = recommend_fertilizers(
#     N=90, P=42, K=43,
#     temperature=20.8,
#     humidity=66.42,
#     ph=8.89,
#     rainfall=20,
#     crop="rice"
# )

# print(response)