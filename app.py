import os
import pickle

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# --- Load model(s) ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} tidak ditemukan. Pastikan file ada.")

with open(MODEL_PATH, "rb") as f:
    loaded = pickle.load(f)

# model bisa single estimator atau list
if isinstance(loaded, (list, tuple)):
    models = list(loaded)
else:
    models = [loaded]


# Buat label yang ramah manusia berdasarkan kelas estimator
def pretty_name(estimator):
    cls = estimator.__class__.__name__
    if cls == "DecisionTreeClassifier":
        return "Decision Tree"
    if cls == "SVC":
        return "SVC"
    # fallback: ubah camelcase jadi spaced words (sederhana)
    # DecisionTree -> Decision Tree, RandomForestClassifier -> RandomForestClassifier (as is)
    import re

    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", cls)
    return s


model_display_names = [pretty_name(m) for m in models]

# --- Load scaler ---
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"{SCALER_PATH} tidak ditemukan. Pastikan file ada.")

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# --- Determine feature names expected by scaler ---
if hasattr(scaler, "feature_names_in_"):
    feature_names = list(scaler.feature_names_in_)
else:
    # fallback sesuai training awalmu
    feature_names = ["lifeExp", "gdpPercap"]


@app.route("/")
def index():
    # kirim model_display_names dan feature_names supaya template menampilkan label benar
    return render_template(
        "index.html",
        model_display_names=model_display_names,
        feature_names=feature_names,
    )


@app.route("/predict", methods=["POST"])
def predict():
    # Model dipilih sebagai index (value option adalah index)
    sel_model_idx_raw = request.form.get("model")
    try:
        sel_model_idx = int(sel_model_idx_raw)
    except Exception:
        sel_model_idx = 0  # fallback

    # ambil input dari form berdasarkan feature_names yang diharapkan
    input_values = {}
    for feat in feature_names:
        raw = request.form.get(feat)
        if raw is None or raw.strip() == "":
            return render_template(
                "index.html",
                model_display_names=model_display_names,
                feature_names=feature_names,
                prediction=f"Missing input for '{feat}'. Silakan isi semua field.",
            )
        try:
            input_values[feat] = float(raw)
        except ValueError:
            return render_template(
                "index.html",
                model_display_names=model_display_names,
                feature_names=feature_names,
                prediction=f"Input untuk '{feat}' harus berupa angka.",
            )

    # buat DataFrame dengan urutan kolom yang sama seperti feature_names
    X_df = pd.DataFrame([input_values], columns=feature_names)

    # transform
    try:
        X_scaled = scaler.transform(X_df)
    except Exception as e:
        return render_template(
            "index.html",
            model_display_names=model_display_names,
            feature_names=feature_names,
            prediction=f"Error saat mengaplikasikan scaler: {e}",
        )

    # pilih model
    sel_idx = sel_model_idx if 0 <= sel_model_idx < len(models) else 0
    model_chosen = models[sel_idx]

    # prediksi
    try:
        y_pred = model_chosen.predict(X_scaled)
        result = y_pred[0] if hasattr(y_pred, "__iter__") else y_pred
    except Exception as e:
        return render_template(
            "index.html",
            model_display_names=model_display_names,
            feature_names=feature_names,
            prediction=f"Error saat prediksi: {e}",
        )

    # tampilkan nama model yang dipakai juga
    used_model_name = model_display_names[sel_idx]
    return render_template(
        "index.html",
        model_display_names=model_display_names,
        feature_names=feature_names,
        prediction=f"Model: {used_model_name} → Prediction: {result}",
        input_values=input_values,
    )


if __name__ == "__main__":
    app.run(debug=True)
