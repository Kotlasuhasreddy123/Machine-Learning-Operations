# ###########################################################################
# Template code for serving MLFlow model from a REST API built with Flask
# ###########################################################################
from flask import Flask, request, jsonify
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

# application 
app = Flask(__name__)

MLFLOW_URI = "http://10.0.0.80:5000"     # modify this with your VM's IP address

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient(tracking_uri=MLFLOW_URI)

# load the model 
model = mlflow.sklearn.load_model(f"models:/suhas_part3_random_forest_model/None")  # Provide the name of your MLFlow model.   You can also specify a version number

#################################
# REST API for HTTP POST
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expecting JSON input with a list of records
        input_json = request.get_json()
        input_data = pd.DataFrame(input_json)

        # Make prediction
        predictions = model.predict(input_data)

        #print the result
        print("Prediction :  ", predictions)

        # Return predictions as JSON
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=57700)