from flask import Flask, render_template, request, Response
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import json
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence

app = Flask(__name__)


# Set up route
# Decoupled root route. Application logic is now handled in POST /predict requests

@app.route('/',  methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    compas_train_data = pd.read_csv('data/upsampled_race_train_dataset.csv')

    compas_test_data = pd.read_csv('data/compas_test_dataset.csv')

    y_train = compas_train_data.pop('two_year_recid').values
    X_train = compas_train_data.values

    y_test = compas_test_data.pop('two_year_recid').values
    X_test = compas_test_data.values

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)

    threshold = float(request.json["threshold"])
    pareto_index = int(request.json["pareto_index"])
    sampleWeights = getSampleWeights(X_scaled, pareto_index)

    logReg = LogisticRegression(max_iter=1000, solver='sag', random_state=42, class_weight={0: threshold, 1: 1-threshold}).fit(
        X_scaled, y_train, sample_weight=sampleWeights)

    modelData = {
        'X_train': X_scaled,
        'X_test': X_test,
        'y_test': y_test,
        'logReg': logReg,
        'df': compas_train_data
    }

    featureImportanceAndPrectionResponse = getFeatureImportance(
        request, modelData)

    featureImportance = featureImportanceAndPrectionResponse["featureImportance"]
    prediction = featureImportanceAndPrectionResponse["predictionResults"]
    inputData = featureImportanceAndPrectionResponse["inputData"]

    flowChartandPrediction = getTradeOffs(request, modelData, inputData)
    prediction = flowChartandPrediction["predictionResults"]
    accuracy = flowChartandPrediction["accuracy"]
    flowChart = flowChartandPrediction["flowChartResults"]

    response_dict = {
        "featureImportance": featureImportance,
        "prediction": prediction,
        "accuracy": accuracy,
        "flowChart": flowChart
    }

    return Response(json.dumps(response_dict), status=201, mimetype='application/json')


def getFeatureImportance(request, modelData):
    allFeatures = list(modelData['df'].columns.values)

    # Dict for pairing variable name to output title
    names = {
        'sex': 'Gender',
        'age': 'Age',
        'race': 'Race',
        'juv_fel_count': 'Juvenile Felonies',
        'juv_misd_count': 'Juvenile Misdemeanors',
        'juv_other_count': 'Other Juvenile Convictions',
        'priors_count': 'Prior Crimes Committed',
        'is_recid': 'Reoffender',
        'is_violent_recid': 'Violent Reoffender',
        'age_cat': 'Age Category',
        'c_charge_degree': 'Current Charge Degree',
        'r_charge_degree': 'Prior Charge Degree'
    }

    # Out array
    featureImportanceData = []
    # Input array for prediction
    inputData = np.zeros(len(allFeatures))
    inputFeatures = list(request.json.keys())

    # remove session and store for later (used in line chart)
    inputFeatures.remove("session")
    session = request.json["session"]

    # Cacluate Age Categrory
    age = int(request.json['age'])
    if age < 25:
        age_cat_key = 'age_cat_Less than 25'
    elif age >= 25 and age <= 45:
        age_cat_key = 'age_cat_25 - 45'
    else:
        age_cat_key = 'age_cat_Greater than 45'

    inputFeatures.append(age_cat_key)

    # remove threshold and pareto_index from list
    inputFeatures.remove("threshold")
    threshold = float(request.json["threshold"])
    inputFeatures.remove("pareto_index")

    logReg = modelData['logReg']
    X = modelData['X_train']

    for key in inputFeatures:
        # Get the dummy column  name from form.
        if key == 'r_charge_degree':
            r_charge_degree_key = request.json[key]
            inputValue = 1
            index = allFeatures.index(r_charge_degree_key)
            name = key
        # Convert c_charge_degree_F == 0 (from radio) to be c_charge_degree_M == 1
        elif (key == 'c_charge_degree_F' or key == 'c_charge_degree_M'):
            name = 'c_charge_degree'
            if (int(request.json[key]) == 0):
                continue
        else:
            if key == age_cat_key:
                inputValue = 1
                name = 'age_cat'
            else:
                inputValue = int(request.json[key])
                name = key
            index = allFeatures.index(key)

        inputData[index] = inputValue

        pdp, axes = partial_dependence(logReg, X=X, features=[index])
        importance = np.interp(inputValue, axes[0], pdp[0])

        featureImportanceData.append({
            "name": names[name],
            "featureImportance": importance,
            "session": session
        })

    outputData = json.dumps(featureImportanceData)

    prediction = logReg.predict([inputData])

    response_dict = {
        "featureImportance": json.loads(outputData),
        "predictionResults": str(prediction[0]),
        "inputData": [inputData]
    }

    return response_dict


def getTradeOffs(request, modelData, inputData):

    threshold = float(request.json["threshold"])
    pareto_index = int(request.json["pareto_index"])

    X_train = modelData['X_train']
    X_test = modelData['X_test']
    y_test = modelData['y_test']
    logReg = modelData['logReg']

    # The first column is the probability that target=0 and the second column is the probability that target=1[:,1]
    y_pred_proba = logReg.predict_proba(X_test)[:, 1]

    FP = 0
    TP = 0
    FN = 0
    TN = 0

    for i in range(len(y_pred_proba)):
        if (y_pred_proba[i] > threshold):
            if y_test[i] == 0:
                FP = FP + 1
            if y_test[i] == 1:
                TP = TP + 1
        else:
            if y_test[i] == 1:
                FN = FN + 1
            if y_test[i] == 0:
                TN = TN + 1

    # Google Flow chart doesn't like rows with 0 values.
    results = []
    if TP != 0:
        results.append(['Labeled will Reoffend', 'Acutally Reoffended', TP])
    if FP != 0:
        results.append(['Labeled will Reoffend', 'Did not Reoffended', FP])
    if TN != 0:
        results.append(['Labeled will not Reoffend', 'Did not Reoffended', TN])
    if FN != 0:
        results.append(['Labeled will not Reoffend',
                        'Acutally Reoffended', FN])

    prediction_proba = logReg.predict_proba(inputData)[:, 1]

    if prediction_proba > threshold:
        prediction = 1
    else:
        prediction = 0

    accuracy = ((TP+TN)/(TP+TN+FN+FP))*100

    return {
        "flowChartResults": results,
        "predictionResults": prediction,
        "accuracy": accuracy
    }


def getSampleWeights(X, pareto_index):
    pareto_front = pd.read_csv('data/pareto_front_dis_total_error.csv')

    theta0 = pareto_front["theta0"][pareto_index]
    theta1 = pareto_front["theta1"][pareto_index]

    sampleWeights = []
    for row in X:

        a = row[23]
        z = 1

        if (theta1 >= theta0):
            z = 1

        c0 = 1*(z == 1) + theta1*(1*(z == 1 and a == 1)) - \
            1*(z == 1 and a == 0)
        c1 = 1*(z == 0) + theta0*(1*(z == 0 and a == 1)) - \
            1*(z == 0 and a == 0)

        if (c0 < c1):
            z = 0
            c0 = 1*(z == 1) + theta1*(1*(z == 1 and a == 1)) - \
                1*(z == 1 and a == 0)
            c1 = 1*(z == 0) + theta0*(1*(z == 0 and a == 1)) - \
                1*(z == 0 and a == 0)

        weight = abs(c0 - c1)
        sampleWeights.append(weight)

    return sampleWeights


if __name__ == "__main__":
    app.run(debug=True)
