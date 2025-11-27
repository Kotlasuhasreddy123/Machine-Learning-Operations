from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score,f1_score,roc_auc_score

from datetime import datetime
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
import psycopg2
from sqlalchemy import create_engine
from joblib import dump, load
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# name of the model file
model_filename = "/tmp/iris_model.pickle"


dag = DAG(                                                    
   dag_id="airflow-ml-pipeline1",                      
   start_date=datetime(2025, 11, 5),               
   schedule=None,  
   catchup=False,                                 
)

# ######################################
def _get_data(ti):
    iris = datasets.load_iris()
    iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_train['Iris type'] = iris['target']
    iris_train['Iris name'] = iris_train['Iris type'].apply(lambda x: 'sentosa' if x == 0 else ('versicolor' if x == 1 else 'virginica'))
    iris_train.head()

    ti.xcom_push(key='iris_data', value=iris_train)    
    return iris_train

# ######################################
def _train_model(ti):
    iris_train = ti.xcom_pull(key='iris_data', task_ids="get_data")

    X = iris_train[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']]
    y = iris_train['Iris name']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    # Note: Model is KNeighborsClassifier in the sample code.
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # save the model 
    dump(model, model_filename)

    # save values for next function
    ti.xcom_push(key='X_train', value=X_train)    
    ti.xcom_push(key='X_test', value=X_test)    
    # y_test is a Series and cannot be serialized. Therefore convert that into a list.
    ti.xcom_push(key='y_test', value=y_test.to_list())    
    return X_train, X_test, y_test.to_list()


# ######################################
def _load_and_predict_model(ti):
    # load the saved model 
    model = load(model_filename)

    # get the X data
    X_train = ti.xcom_pull(key='X_train', task_ids="train_model")
    X_test = ti.xcom_pull(key='X_test', task_ids="train_model")
    y_test = ti.xcom_pull(key='y_test', task_ids="train_model")

    # run the prediction
    y_pred_rfc = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred_rfc)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    precision = precision_score(y_test, y_pred_rfc, average='weighted')
    print(f'Precision : {precision * 100:.2f}%')
    f1 = f1_score(y_test, y_pred_rfc, average='weighted')
    print(f'F1 : {f1 * 100:.2f}%')

    # save values for next function
    ti.xcom_push(key='X_train', value=X_train)    
    ti.xcom_push(key='accuracy', value=accuracy)    
    ti.xcom_push(key='precision', value=precision)    
    ti.xcom_push(key='f1', value=f1)

    return X_train, accuracy, precision, f1

# ######################################
def _log_model_in_mlflow(ti):
    # load the saved model 
    model = load(model_filename)

    # get the X data
    X_train = ti.xcom_pull(key='X_train',      task_ids="load_and_predict_model")
    accuracy = ti.xcom_pull(key='accuracy',    task_ids="load_and_predict_model")
    precision = ti.xcom_pull(key='precision',  task_ids="load_and_predict_model")
    f1 = ti.xcom_pull(key='f1',                task_ids="load_and_predict_model")


    # =======================================================
    # !!! MODIFICATIONS FOR PART 2 !!!
    # REPLACE 'lewisu' WITH YOUR NAME/ID
    # =======================================================
    exp_name = "suhas_part2_experiment"    
    model_name = "suhas_part2_model"       
    url = "http://10.0.0.80:5000"                 # YOUR VM IP ADDRESS

    # set the MLFlow tracking URI
    mlflow.set_tracking_uri(uri=url)

    # set the experiment name
    mlflow.set_experiment(exp_name)
    with mlflow.start_run():
      signature = infer_signature(X_train, model.predict(X_train))

      # log the model
      mlflow.sklearn.log_model(model, name=model_name, registered_model_name=model_name, signature=signature)

      # log the metrics
      mlflow.log_metric("Accuracy", accuracy)
      mlflow.log_metric("Precision", precision)
      mlflow.log_metric("F1", f1)

      mlflow.end_run()
    # =======================================================

################################################
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