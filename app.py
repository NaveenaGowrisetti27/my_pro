from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

saved_objects = joblib.load("model.pkl")

model = saved_objects["model"]
encoder = saved_objects["encoder"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["sepal_length"]),
            float(request.form["sepal_width"]),
            float(request.form["petal_length"]),
            float(request.form["petal_width"])
        ]

        prediction = model.predict([features])[0]
        species = encoder.inverse_transform([prediction])[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Species: {species}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)