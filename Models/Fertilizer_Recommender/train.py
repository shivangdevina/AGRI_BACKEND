import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from datasets import load_dataset
from utils import save_model, save_json

print("📥 Loading dataset...")
dataset = load_dataset("Jakehills/Crop_Yield_Fertilizer")
df = dataset["train"].to_pandas()

# Features & target
X = pd.get_dummies(df.drop("fertilizer", axis=1), columns=["label"])
le = LabelEncoder()
y = le.fit_transform(df["fertilizer"])

# Train XGBoost
print("⚡ Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss"
)
xgb_model.fit(X, y)

# Train Random Forest
print("🌲 Training RandomForest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X, y)

# Ensemble
print("🤝 Building ensemble...")
ensemble_model = VotingClassifier(
    estimators=[("rf", rf_model), ("xgb", xgb_model)],
    voting="soft"
)
ensemble_model.fit(X, y)

# Save
save_model(xgb_model, "xgb_model.pkl")
save_model(rf_model, "rf_model.pkl")
save_model(ensemble_model, "ensemble_model.pkl")
save_model(le, "label_encoder.pkl")
save_json(list(X.columns), "feature_columns.json")

print("✅ Models trained and saved in /models/")
