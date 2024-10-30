from flask import Flask, render_template, request
import pickle as pkl
# from sklearn.ensemble import RandomForestClassifier
import numpy as np


# Flask App Here
app = Flask(__name__)

# Home Page
@app.route('/')
def home():
    return render_template("index.html")

def getData():
    gender = int(request.form.get("gender"))
    age = int(request.form.get("age"))
    smoke = int(request.form.get("smoke"))
    yellow_finger = int(request.form.get("yellow_finger"))
    anxiety = int(request.form.get("anxiety"))
    peer_pressure = int(request.form.get("peer_pressure"))
    chronic_disease = int(request.form.get("chronic_disease"))
    fatigue = int(request.form.get("fatigue"))
    allergy = int(request.form.get("allergy"))
    wheezing = int(request.form.get("wheezing"))
    alcohol = int(request.form.get("alcohol"))
    cough = int(request.form.get("cough"))
    shortness_in_breath = int(request.form.get("shortness_in_breath"))
    swallowing = int(request.form.get("swallowing"))
    chest_pain = int(request.form.get("chest_pain"))

    X_pred = [[gender, age, smoke, yellow_finger, anxiety,
                peer_pressure, chronic_disease, fatigue,
               allergy, wheezing, alcohol, cough, shortness_in_breath,
               swallowing, chest_pain]]

    print(X_pred)
    return X_pred

def modelPredict(X_pred):
    with open("/Users/sakibdalal/Downloads/Lung-Cancer-Prediction/models/LogisticRegressionModel.pkl", "rb") as f:
        model = pkl.load(f)
    y_pred = model.predict(X_pred)
    print(y_pred)
    return y_pred

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        data = getData()
        output = int(modelPredict(data)[0])
        if output == 1:
            return "<h1>You have lung cancer</h1>"
        else:
            return "<h1>You don't have lung cancer</h1>"
    except ValueError:
        return render_template("index.html") 
   

if __name__ == "__main__":
    app.run(debug=True, port="80", host="0.0.0.0")