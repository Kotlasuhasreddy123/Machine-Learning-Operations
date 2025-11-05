# Final Airflow DAG Code for Incremental Load
# =============================================================

from datetime import datetime
import pandas as pd
# Airflow specific imports
from airflow import DAG
from airflow.operators.python import PythonOperator

# Database connection libraries
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# --- CRITICAL DATABASE CONNECTION CONFIGURATION ---
# These parameters use the successful external IP and port/DB name setup:
HOST_IP = '10.0.0.80' 
DB_PORT = '5433' 
DB_NAME = 'postgres' # The database name you created in the final steps
DB_USER = 'postgres'
DB_PASSWORD = 'password'
TABLE_NAME = 'airline_weather_data'

# Define the DAG
dag = DAG(                                                    
   dag_id="incremental_load_pipeline",          
   start_date=datetime(2025, 11, 1),      
   schedule=None, 
   catchup=False,
   default_args={
       'owner': 'lewisu',
       'retries': 0, 
   }                                 
)

##############################
# Task 1: Read CSV, Transform Data, and Push to XCom
##############################
def _read_csv_data(ti):
    # Airflow automatically mounts the data volume to /opt/airflow/data
    file_path = "/opt/airflow/data/incremental-data.csv"
    
    try:
        # Uses your proven read logic
        airline_df = pd.read_csv(file_path, sep=',', on_bad_lines='skip')
    except FileNotFoundError:
        print(f"File not found at {file_path}. Aborting.")
        raise
    
    # Feature transformation (Your existing logic)
    airline_df['CrewAvailable'] = airline_df['CrewAvailable'].apply(lambda x: 0 if x == 'N' else 1)
    airline_df['BusyRunways'] = airline_df['BusyRunways'].apply(lambda x: 0 if x == 'N' else 1)
    airline_df['DelayedYN'] = airline_df['DelayedYN'].apply(lambda x: 0 if x == 'N' else 1)
    
    print(f"Successfully read and transformed {len(airline_df)} rows.")

    # PUSH to XCom: Send the data out of this task for the next task to retrieve
    ti.xcom_push(key='airline_key', value=airline_df.to_dict('records')) 
    return True 

##############################
# Task 2: Pull Data from XCom and Write to PostgreSQL
##############################
def _write_data_to_DB(ti):
    # PULL from XCom: Retrieve the data from the previous task
    df_data = ti.xcom_pull(task_ids='read_csv_data', key='airline_key')
    
    if not df_data:
        print("No data found in XCom. Aborting database write.")
        return

    # Reconstruct DataFrame from the list of dictionaries pulled from XCom
    airline_df = pd.DataFrame(df_data)

    # 1. Build the SQLAlchemy connection string
    conn_string = f'postgresql://{DB_USER}:{DB_PASSWORD}@{HOST_IP}:{DB_PORT}/{DB_NAME}'
    
    # CRITICAL NETWORK TEST (Your proven connection logic)
    try:
        conn_psycopg = psycopg2.connect(
            host=HOST_IP, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        conn_psycopg.close()
        print(f"CONNECTION TEST SUCCESSFUL: Airflow worker can reach {HOST_IP}:{DB_PORT} / DB: {DB_NAME}.")
    except Exception as e:
        print("=========================================================================================")
        print(f"CRITICAL NETWORK FAILURE: Cannot connect to the DB. Error: {e}")
        print("=========================================================================================")
        raise # Force failure

    try:
        # 2. Establish Engine
        db = create_engine(conn_string)
        
        # 3. Use 'with db.begin()' to manage the transaction
        with db.begin() as conn:  
            
            # 4. Use to_sql with if_exists='append' for incremental loading (the critical step for Part 5)
            print(f"Appending {len(airline_df)} rows to '{TABLE_NAME}'...")
            airline_df.to_sql(TABLE_NAME, conn, if_exists='append', index=False)  
            
            print(f"Successfully appended {len(airline_df)} rows into '{TABLE_NAME}'.")
        
    except SQLAlchemyError as e:
        print(f"DATABASE WRITE ERROR: Failed during data insertion. Error: {e}")
        raise

# Define the tasks
read_csv_data = PythonOperator(
    task_id='read_csv_data',
    python_callable=_read_csv_data,
    dag=dag,
)

write_data_to_DB = PythonOperator(
    task_id='write_data_to_DB',
    python_callable=_write_data_to_DB,
    dag=dag,
)

# Set the task dependencies: Read CSV must finish before writing to DB starts
read_csv_data >> write_data_to_DB