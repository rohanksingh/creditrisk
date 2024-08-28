# Model Training and Validation 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from creditrisk import create_server_connection

# Database connection details
pw = "sUmitra@12"
db = "sakila"


connection = create_server_connection("localhost", "root", pw, db)

# Check if the connection was successful before proceeding
if connection is not None:
    query = "SELECT * FROM sakila.credit_data_1"
    data = pd.read_sql(query, connection)
    print(data.head())  # Display the first few rows of the data
else:
    print("Connection failed.")


# Missing value and outliers
data.loc[data.sample(frac=0.1).index, 'income']= np.nan
data.loc[data.sample(frac=0.05).index, 'loan_amount'] *=-1

# Display the first few rows of the Dataframe
print(data.head())

# Data Preprocessing (filling missing values and handling outliers)
data.fillna(method='ffill' , inplace= True)
data['loan_amount']= data['loan_amount'].apply(lambda x:x if x>0 else None)


# Feature selection 
features =['credit_score', 'income', 'loan_amount', 'loan_duration']
X= data[features]
y= data['default']

# # Split data 
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=42)

# Model training 

model= RandomForestClassifier()
model.fit(X_train, y_train)

# Model validation

y_pred= model.predict_proba(X_test)[:,1]
auc_score= roc_auc_score(y_test, y_pred)

print(f"AUC Score: {auc_score:.2f}")

import schedule
import time

def generate_risk_report():

    risk_scores= model.predict_proba(X_test)[:,1]
    report=pd.DataFrame({'Risk Score': risk_scores})
    report.to_csv('risk_report.csv', index=False)
    print("Risk report generated")

schedule.every().day.at("09:00").do(generate_risk_report)

while True:
    schedule.run_pending()
    time.sleep(1)

import numpy as np

stress_scenarios= {
    'High Unemploymnet' : {'unemployment_rate': 0.12},
    'Economic Downturn': {'gdp_growth': -0.02},
}

for scenario, changes in stress_scenarios.items():
    stressed_data= data.copy()
    for feature, change in changes.items():
        stressed_data[features]*= change
    stressed_scores= model.predict_proba(stressed_data[features])[:,1]
    print(f"Average Risk Score under {scenario}: {np.mean(stressed_scores):.2f}")