from flask import Flask, render_template, request
import numpy as np
import joblib
import tensorflow as tf
import xgboost as xgb
import os

app = Flask(__name__)

BASE_DIR = r"E:\water_quality_app\models"
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
LSTM_PATH = os.path.join(BASE_DIR, "lstm_water_quality.h5")
XGB_PATH = os.path.join(BASE_DIR, "xgboost_water_quality.json")
META_PATH = os.path.join(BASE_DIR, "meta_logistic.pkl")

print("🔄 Loading models...")
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)
meta_model = joblib.load(META_PATH)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_PATH)

print("🔄 Loading LSTM model...")
lstm_model = None
try:
    if os.path.exists(LSTM_PATH):
        lstm_model = tf.keras.models.load_model(LSTM_PATH, compile=False)
        print("✅ LSTM model loaded successfully.")
    else:
        print(f"⚠️ LSTM model not found at: {LSTM_PATH}")
except Exception as e:
    print(f"⚠️ Could not load LSTM model: {e}")
    lstm_model = None

print("✅ All models loaded successfully!\n")

FEATURES = ["temp", "do", "ph", "cond", "bod", "nitrate"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect year separately
        year = request.form.get("year")

        user_input = []
        for f in FEATURES:
            val = request.form.get(f)
            if val is None or val.strip() == "":
                return render_template("result.html", error=f"Missing input: {f}")
            user_input.append(float(val))

        input_array = np.array(user_input).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        xgb_pred_proba = xgb_model.predict_proba(scaled_input)[0]

        if lstm_model is not None:
            lstm_input = scaled_input.reshape((scaled_input.shape[0], 1, scaled_input.shape[1]))
            lstm_pred_proba = lstm_model.predict(lstm_input, verbose=0)[0]
        else:
            lstm_pred_proba = np.zeros_like(xgb_pred_proba)

        stacked_input = np.concatenate([xgb_pred_proba, lstm_pred_proba]).reshape(1, -1)
        meta_pred = meta_model.predict(stacked_input)
        meta_pred_label = label_encoder.inverse_transform(meta_pred)[0]

        return render_template("result.html", prediction=meta_pred_label, year=year)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template("result.html", error=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
