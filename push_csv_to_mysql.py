import pandas as pd
import mysql.connector
from mysql.connector import Error



def create_server_connection(host_name, user_name, user_password, db_name):
    connection= None

    try:
        connection =mysql.connector.connect(
            host= host_name,
            user= user_name,
            passwd= user_password,
            database=db_name
            
        )
        print("MYSQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")
    return connection


def push_csv_to_mysql(csv_file_path, connection):
    
    try:

        df= pd.read_csv(csv_file_path, encoding= 'utf-8-sig')

        cursor= connection.cursor()

        insert_query= """
        INSERT INTO creditdata (customer_id	,credit_score, income,	loan_amount,	loan_duration,	'default')
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        for index, row in df.iterrows():
            cursor.execute(insert_query, (row['customer_id'], 
                                          row['credit_score'], 
                                          row['income'], 
                                          row['loan_amount'], 
                                          row['loan_duration'], 
                                          row['default']))
            
        connection.commit()
            
        print("Data inserted successfully")

    except Error as err:
        print(f"Error: '{err}'")
        
    finally:
        
        cursor.close()

# Database connection details
pw = "sUmitra@12"
db = "sakila"

# Establish the connection
connection = create_server_connection("localhost", "root", pw, db)

# Path to your CSV file
csv_file_path = 'C:/Users/rohan/CreditRisk/credit_data.csv'

# Push CSV data to MySQL
if connection:
    push_csv_to_mysql(csv_file_path, connection)
    connection.close()
else:
    print("Connection failed.")




