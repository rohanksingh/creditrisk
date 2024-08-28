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

if __name__ == "__main__":
    pw= "sUmitra@12"
    db= "sakila"
    connection= create_server_connection("localhost", "root", pw, db)


# Check if the connection was successful before processing 

# if connection is not None:
#     query ="SELECT * FROM sakila.credit_data"
#     data= pd.read_sql(query, connection)
#     print(data.head())
# else:
#     print("Connection failed.")


# # Data Preprocessing 
# data.fillna(method= 'ffill', inplace=True)
# data['loan_amount']= data['loan_amount'].apply(lambda x:x if x>0 else None)

