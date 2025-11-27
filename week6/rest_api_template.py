import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

import mlflow
from mlflow.models import infer_signature
from mlflow import MlflowClient

MLFLOW_URI = "http://10.0.0.80:5000"
mlflow.set_tracking_uri(MLFLOW_URI)

mlflow.set_experiment("Week6_Suhas_Experiments")
client = MlflowClient(tracking_uri=MLFLOW_URI)
REGISTERED_MODEL_NAME = "suhas_week6_model"

def train_and_log_model(n_estimators_param, max_depth_param, run_tag_value, algorithm_name):
    if "RandomForest" in algorithm_name:
        model_class = RandomForestClassifier
        model_params = {'n_estimators': n_estimators_param, 'max_depth': max_depth_param, 'random_state': 42}
    elif "DecisionTree" in algorithm_name:
        model_class = DecisionTreeClassifier
        model_params = {'max_depth': max_depth_param, 'random_state': 42}
    else:
        print(f"Algorithm {algorithm_name} not recognized. Skipping run.")
        return

    with mlflow.start_run(run_name=f"{algorithm_name}_n{n_estimators_param}_d{max_depth_param}") as run:
        
        data = pd.read_csv("/home/lewisu/airline-historical-data.csv")
        
        # Data Cleaning and Feature Engineering
        data['Weather'] = data['Weather'].astype(str).str.lower()
        data['CrewAvailable'] = data['CrewAvailable'].astype(str).str.lower()
        
        data = pd.get_dummies(data, columns=['Weather'], prefix='Weather')
        data = pd.get_dummies(data, columns=['CrewAvailable'], prefix='CrewAvailable')
        
        # FIX for 'ValueError: could not convert string to float'
        # Coerce non-dummy features to numeric, replacing bad strings with 0
        data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce').fillna(0)
        data['BusyRunways'] = pd.to_numeric(data['BusyRunways'], errors='coerce').fillna(0)
        
        le = LabelEncoder()
        data['Delay'] = le.fit_transform(data['DelayedYN'])
        
        features = [
            'Duration', 
            'BusyRunways',
            'Weather_sunny',      
            'CrewAvailable_y'     
        ]
        target = 'Delay' 

        X = data[features] 
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = model_class(**model_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"Algorithm: {algorithm_name}, Accuracy: {accuracy:.4f}")

        mlflow.log_param("algorithm", algorithm_name)
        for param, value in model_params.items():
            mlflow.log_param(param, value)
        mlflow.log_param("split_ratio", 0.8)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.set_tag("model_type", "Classification")
        mlflow.set_tag("important_feature", run_tag_value)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=infer_signature(X_train, model.predict(X_train))
        )
        
if __name__ == '__main__':
    train_and_log_model(n_estimators_param=100, max_depth_param=8, run_tag_value="Weather_sunny", algorithm_name="RandomForest_100")
    
    train_and_log_model(n_estimators_param=100, max_depth_param=15, run_tag_value="Crew_Available", algorithm_name="RandomForest_15d")

    train_and_log_model(n_estimators_param=0, max_depth_param=5, run_tag_value="Fast_Tree", algorithm_name="DecisionTree_5d")

    train_and_log_model(n_estimators_param=300, max_depth_param=8, run_tag_value="All_Features", algorithm_name="RandomForest_300")