from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score

from datetime import datetime
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from joblib import dump, load
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# name of the model file
model_filename = "/tmp/airline_delay_model.pickle"
DATA_FILENAME = "airline-historical-data.csv"
DATA_PATH = f"/opt/airflow/data/{DATA_FILENAME}"

dag = DAG(                                                    
   dag_id="airflow-ml-pipeline-part3-airline",                 
   start_date=datetime(2025, 1, 1),               
   schedule=None,  
   catchup=False,                                 
)

def _get_data(ti):
    # DAG Step 1: Read the airline test data CSV file from the /opt/airflow/data folder
    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: CSV file not found. Ensure '{DATA_FILENAME}' is in /opt/airflow/data")
        raise
        
    # Preprocessing: Convert target to binary (1 for Yes, 0 for No)
    data['is_delayed'] = data['DelayedYN'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Preprocessing: Select features and perform one-hot encoding for the model
    # Features: numerical + categorical conditions
    features_num = ['MM', 'DD', 'YY', 'Duration']
    features_cat = ['Weather', 'CrewAvailable', 'BusyRunways']
    
    # Select only the relevant columns
    data = data[features_num + features_cat + ['is_delayed']]
    
    # One-Hot Encoding for categorical features
    data_processed = pd.get_dummies(data, columns=features_cat, drop_first=True)
    
    # Push the processed DataFrame (as JSON) into XCOM
    ti.xcom_push(key='airline_data_processed', value=data_processed.to_json())    
    return data_processed.to_json()

def _train_model(ti):
    # DAG Step 2: Get data from XCOM
    airline_data_json = ti.xcom_pull(key='airline_data_processed', task_ids="get_data")
    airline_data = pd.read_json(airline_data_json)

    target = 'is_delayed'
    
    # X includes all columns except the target
    X = airline_data.drop(columns=[target])
    y = airline_data[target]
    
    # Create the X and y using the train-test-split function
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y) 

    # Use the RandomForestClassifier algorithm
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train) 

    # Save the model into a pickle file
    dump(model, model_filename)

    # Push the test data for the next step
    ti.xcom_push(key='X_test', value=X_test.to_json())    
    ti.xcom_push(key='y_test', value=y_test.to_list())
    ti.xcom_push(key='X_train_cols', value=X_train.columns.to_list()) # Save columns for signature inference
    
    return X_test.to_json(), y_test.to_list()


def _load_and_predict_model(ti):
    # DAG step 3: Load the saved model and run a prediction
    model = load(model_filename)

    # Get the X and y data from XCOM
    X_test_json = ti.xcom_pull(key='X_test', task_ids="train_model")
    y_test = ti.xcom_pull(key='y_test', task_ids="train_model")
    X_test = pd.read_json(X_test_json)

    # Run the predict function
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability for ROC AUC

    # Compute the Accuracy, Precision, F1 and Roc-auc metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision : {precision * 100:.2f}%')
    print(f'F1 : {f1 * 100:.2f}%')
    print(f'ROC AUC: {roc_auc * 100:.2f}%')

    # Push the metrics into XCOM for the next step
    ti.xcom_push(key='X_test', value=X_test_json) 
    ti.xcom_push(key='accuracy', value=accuracy)    
    ti.xcom_push(key='precision', value=precision)    
    ti.xcom_push(key='f1', value=f1)
    ti.xcom_push(key='roc_auc', value=roc_auc) 

    return accuracy, precision, f1, roc_auc

def _log_model_in_mlflow(ti):
    # DAG Step 4: Log the model in MLFlow
    model = load(model_filename)

    # Get the data and metrics from XCOM
    X_test_json = ti.xcom_pull(key='X_test',      task_ids="load_and_predict_model")
    X_test = pd.read_json(X_test_json)
    X_train_cols = ti.xcom_pull(key='X_train_cols', task_ids="train_model")
    X_test.columns = X_train_cols # Ensure columns match for signature
    
    accuracy = ti.xcom_pull(key='accuracy',    task_ids="load_and_predict_model")
    precision = ti.xcom_pull(key='precision',  task_ids="load_and_predict_model")
    f1 = ti.xcom_pull(key='f1',                task_ids="load_and_predict_model")
    roc_auc = ti.xcom_pull(key='roc_auc',      task_ids="load_and_predict_model")


    # Set the MLFlow tracking URL
    url = "http://10.0.0.80:5000"                
    mlflow.set_tracking_uri(uri=url)

    # Set the MLFlow experiment and model names
    exp_name = "suhas_part3_airline_experiment"    
    model_name = "suhas_part3_random_forest_model"      
    mlflow.set_experiment(exp_name)
    
    with mlflow.start_run():
      # Infer signature using test data
      signature = infer_signature(X_test.head(5), model.predict(X_test.head(5))) 

      # Log the model and the metrics in MLFlow
      mlflow.sklearn.log_model(model, name=model_name, registered_model_name=model_name, signature=signature)

      # Log all four required metrics
      mlflow.log_metric("Accuracy", accuracy)
      mlflow.log_metric("Precision", precision)
      mlflow.log_metric("F1", f1)
      mlflow.log_metric("Roc-auc", roc_auc) 

      mlflow.end_run()

# configure the tasks

get_data = PythonOperator(                                
   task_id="get_data",
   python_callable=_get_data,                             
   dag=dag,
)

train_model = PythonOperator(                                
   task_id="train_model",
   python_callable=_train_model,                             
   dag=dag,
)

load_and_predict_model = PythonOperator(                                
   task_id="load_and_predict_model",
   python_callable=_load_and_predict_model,                             
   dag=dag,
)

log_model_in_mlflow = PythonOperator(                                
   task_id="log_model_in_mlflow",
   python_callable=_log_model_in_mlflow,                             
   dag=dag,
)

# specify the pipeline steps
get_data >> train_model >> load_and_predict_model >> log_model_in_mlflow