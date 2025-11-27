# DAG to poll a directory and load data from CSV files that show up in that directory
# =============================================================

from datetime import datetime
import pandas as pd
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
import psycopg2
from sqlalchemy import create_engine

# --- CRITICAL DATABASE CONNECTION CONFIGURATION ---
HOST_IP = '10.0.0.80' 
DB_PORT = '5433' 
DB_NAME = 'postgres' 
DB_USER = 'postgres'
DB_PASSWORD = 'password'
TABLE_NAME = 'airline_weather_data'
FILE_PATH = "/opt/airflow/data/incremental-data.csv" # Location of the file to watch

# DAG Definition: Set to continuously check for the file
dag = DAG(                                                    
   dag_id="file_watcher_incremental_load",       # New, unique DAG name          
   start_date=datetime(2025, 11, 1),
   schedule="@continuous",
   max_active_runs=1,    
   catchup=False,                                 
)

##############################
# Task 1: Wait for a file to show up
##############################
# NOTE: You must create an Airflow File Connection named 'FileWatcher'
# that points to your data folder (/opt/airflow/data)
wait_for_file = FileSensor(
    task_id = "wait_for_file",
    filepath=FILE_PATH,     
    fs_conn_id='FileWatcher', # Requires creating a File Connection in Airflow UI
    poke_interval=5, # Check every 5 seconds
    timeout=600, # Wait for a maximum of 10 minutes
    dag=dag,
)

##############################
# Task 2: Read CSV, Transform Data, and Push to XCom
##############################
def _read_csv_data(ti):
    
    try:
        airline_df = pd.read_csv(FILE_PATH, sep=',', on_bad_lines='skip')
        
        # --- FIX FOR KEYERROR ---
        # Normalize column names by stripping whitespace and lowercasing
        airline_df.columns = airline_df.columns.str.strip().str.lower()
        # ------------------------

    except FileNotFoundError:
        print(f"File not found at {FILE_PATH}. Aborting.")
        raise
    
    # Feature transformation (Updated to use lowercase column names)
    # Corrected 'busyruntways' to 'busyrunways'
    airline_df['crewavailable'] = airline_df['crewavailable'].apply(lambda x: 0 if x == 'N' else 1)
    airline_df['busyrunways'] = airline_df['busyrunways'].apply(lambda x: 0 if x == 'N' else 1)
    airline_df['delayedyn'] = airline_df['delayedyn'].apply(lambda x: 0 if x == 'N' else 1)
    
    print(f"Successfully read and transformed {len(airline_df)} rows.")

    # PUSH to XCom
    ti.xcom_push(key='airline_key', value=airline_df.to_dict('records')) 
    return True 

##############################
# Task 3: Pull Data from XCom and Write to PostgreSQL
##############################
def _write_data_to_DB(ti):
    df_data = ti.xcom_pull(task_ids='read_csv_data', key='airline_key')
    
    if not df_data:
        print("No data found in XCom. Aborting database write.")
        return

    # Reconstruct DataFrame
    airline_df = pd.DataFrame(df_data)

    # Build the SQLAlchemy connection string
    conn_string = f'postgresql://{DB_USER}:{DB_PASSWORD}@{HOST_IP}:{DB_PORT}/{DB_NAME}'
    
    try:
        # Establish Engine
        db = create_engine(conn_string)
        
        # Use 'with db.begin()' to manage the transaction
        with db.begin() as conn:  
            
            # Use to_sql with if_exists='append' for incremental loading
            print(f"Appending {len(airline_df)} rows to '{TABLE_NAME}'...")
            airline_df.to_sql(TABLE_NAME, conn, if_exists='append', index=False)  
            
            print(f"Successfully appended {len(airline_df)} rows into '{TABLE_NAME}'.")
        
    except Exception as e:
        print(f"DATABASE WRITE ERROR: Failed during data insertion. Error: {e}")
        raise

# Define the tasks
read_csv = PythonOperator(                                
   task_id="read_csv_data",
   python_callable=_read_csv_data,                             
   dag=dag,
)

write_data = PythonOperator(                                
   task_id="write_data_to_DB",
   python_callable=_write_data_to_DB,                             
   dag=dag,
)

##############################
# Task 4: Archive the processed file
##############################
# Note: You need a directory named 'archive' inside your Airflow data folder!
archive_file = BashOperator (
    task_id = "archive_file",
    # This moves the file and adds a unique run ID to the filename
    bash_command = 'mv /opt/airflow/data/incremental-data.csv /opt/airflow/archive/incremental-data-{{ run_id }}.csv'   
)

# Set the task dependencies: Wait -> Read -> Write -> Archive
wait_for_file >> read_csv >> write_data >> archive_file