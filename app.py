import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

# Pemetaan angka ke nama spesies Iris
label_mapping = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    feature = [np.array(float_feature)]
    prediction = model.predict(feature)  # Output: [2] misalnya
    
    # Ubah angka ke nama spesies
    predicted_label = label_mapping.get(prediction[0], "Unknown")

    return render_template("index.html", prediction_text=f"Prediksi: {predicted_label}")

if __name__ == '__main__':
    app.run(debug=True)
