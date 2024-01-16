from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import calibration_curve
import json
import time
import matplotlib.pyplot as plt

app = Flask(__name__)

with open("dados.json", "r") as json_file:
    data = json.load(json_file)

data = pd.DataFrame(data)
data["Outcome"] = data["Outcome"].astype(int)

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mlp = MLPClassifier(
    solver="adam",
    max_iter=1000,
    activation="relu",
    hidden_layer_sizes=(16, 12, 8, 4),
    alpha=0.1,
    batch_size='auto',
    learning_rate="constant",
    learning_rate_init=0.001,
    random_state=2
)

# Medindo o tempo de treinamento
start_time = time.time()
mlp.fit(X_scaled, y)
end_time = time.time()
training_time = end_time - start_time
print(f"Tempo de treinamento: {training_time} segundos")

mlp.fit(X_scaled, y)


def make_prediction(model, features):
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    return prediction[0], probabilities[0]


@app.route('/predict', methods=['POST'])
def predict_diabetes():
    try:
        input_data = request.get_json()
        prediction_results = diagnose_diabetes(input_data)
        return jsonify({"results": prediction_results})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/calibration_curve', methods=['GET'])
def get_calibration_curve():
    try:
        probabilities = mlp.predict_proba(X_scaled)
        diabetes_probabilities = probabilities[:, 1]

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, diabetes_probabilities, n_bins=10)

        calibration_data = {
            "fraction_of_positives": list(fraction_of_positives),
            "mean_predicted_value": list(mean_predicted_value)
        }

        return jsonify(calibration_data)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/accuracy', methods=['GET'])
def get_model_accuracy():
    try:
        accuracy = calculate_accuracy(mlp, X_scaled, y)
        return jsonify({"accuracy": round(float(accuracy), 2)})
    except Exception as e:
        return jsonify({"error": str(e)})


def diagnose_diabetes(patient_data):
    patient_data_df = pd.DataFrame(patient_data, index=[0])

    result = {
        "model": "Multilayer Perceptron (MLP)"
    }

    prediction, probabilities = make_prediction(mlp, patient_data_df)
    probability_diabetes = probabilities[1] * 100
    probability_not_diabetes = probabilities[0] * 100

    accuracy = calculate_accuracy(mlp, X_scaled, y)

    result["prediction"] = int(prediction)
    result["probability_diabetes"] = round(float(probability_diabetes), 2)
    result["probability_not_diabetes"] = round(
        float(probability_not_diabetes), 2)
    result["accuracy"] = round(float(accuracy), 2)

    return [result]


def calculate_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    return accuracy


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
