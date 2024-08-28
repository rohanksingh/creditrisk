import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV , StratifiedKFold
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import scipy.stats as stats
import itertools
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('train.csv')
df.info()
df.head()

df1= pd.DataFrame(df)

df1.columns = df1.columns.str.lower()
df1.info()

# Correcting column names
# df.columns= df.columns.str.lower()

df1.describe().T

cols_to_transform = ['age', 'annual_income', 
                     'num_of_loan', 
                     'num_of_delayed_payment',
                     'changed_credit_limit',
                     'outstanding_debt',
                     #'credit_history_age',
                     'amount_invested_monthly',
                     'monthly_balance']

# columns1= ['id',
# 'customer_id',
# 'month',
# 'name',
# 'age',
# 'ssn',
# 'occupation',
# 'annual_income',
# 'num_of_loan',
# 'type_of_loan',
# 'num_of_delayed_payment',
# 'changed_credit_limit',

# 'credit_mix',
# 'outstanding_debt',

# 'credit_history_age',
# 'payment_of_min_amount',

# 'amount_invested_monthly',
# 'payment_behaviour',
# 'monthly_balance',
# 'credit_score'
# ]


# Some column should be numeric , so it will be modified

# def str_to_num(data, columns):
    
data = df1.copy()
data.info()
for col in cols_to_transform:
    data[col] = data[col].astype(float)
        
  
data.describe().T

for col in cols_to_transform:
    print(f'{col} {data[col].str.contains("_").sum()}')
    
    
# for col in columns1:
#     print(f'{col} {data[col].str.contains("_").sum()}')  


for col in cols_to_transform:
    print(f'{col}: \n\n{data[data[col].astype(str).str.contains("_")][col]}\n')
    
# Removing "_" from data 

for col in cols_to_transform:
    data[col] = data[col].astype(str).str.replace('(_|__)', '', regex=True)

# Validate 

for col in cols_to_transform:
    print(f'{col}: \n\n{data[data[col].astype(str).str.contains("_")][col]}\n')
    
    
# Null values assign

# for col in cols_to_transform:
#     data[col] =data[col].astype(str).apply(lambda x: np.nan if x == '_' else x) 
# data

for col in cols_to_transform:
    data[col] =data[col].astype(str).apply(lambda x: np.nan if x == '' else x)
data
data.info()
       
    
def str_to_num(data, columns):
    
    data= data.copy()
    
    for col in columns:
        data[col] = data[col].astype(float) # checking initally data
        
    return data

data= str_to_num(data, cols_to_transform)
data.info()

#Summary

data.describe().applymap(lambda x: "{:,.2f}".format(x)).T

data.describe().applymap(lambda x: "{:,.2f}".format(x)).T[['mean', 'max']]


# Now let's transform credit age in numbers. 
# We do not have the account opening date, so we are gonna use 365 days for 
# Year and 30 days for Months, to convert credit history age to days.


data['credit_hist_age_Y_to_D'] = data['credit_history_age']\
                                    .str.split('and', expand=True)[0]\
                                        .str.replace(' Years', '')\
                                            .map(lambda x: int(x)*365 if type(x)== str else np.nan)
                                            
data['credit_hist_age_M_to_D'] = data['credit_history_age']\
    .str.split('and', expand= True)[1]\
        .str.replace(' Months', '')\
            .map(lambda x: int(x)*30 if type(x) == str else np.nan)
    
data['credit_history_age_days'] = data['credit_hist_age_Y_to_D'] + data['credit_hist_age_M_to_D']

data['credit_history_age_days']
        
data= data.drop(columns = ['credit_hist_age_Y_to_D', 'credit_hist_age_M_to_D', 'credit_history_age'])

ncols= 2
nrows= 9

fig, ax = plt.subplots(nrows=nrows, ncols= ncols, figsize= (15,30)) 

for i , col in enumerate(data.select_dtypes('number').columns):
    
    idx_col = i // nrows
    idx_row = i % nrows
    
    # print(idx_row, idx_col, col)
    
    sns.boxplot(data[col], ax = ax[idx_row, idx_col])
    ax[idx_row, idx_col].set_title(f'{str.title(col)}')
    
  # Checking the outliers for filling missing values 
  
data.isnull().mean().sort_values(ascending=True).plot(kind='barh')
plt.title('Missing Count') 
            
# maximum amount of missing is 14% so can fill them after working on outliers

map_month ={ 'January' : 1,
            'February' : 2,
            'March' : 3, 
            'April' : 4, 
            'May' : 5,
            'June' : 6,
            'July' : 7,
            'August': 8}

data['month'] =data['month'].map(map_month)
data['month']
data.head()
data.info()
# data.to_csv("data.csv")

# EDA + Feat.Engineering

# feature engineering , 

pd.set_option('display.max_columns', 50)

def get_value_on_max_month(column, month_column):
    return column.loc[month_column.idxmax()]

data_tf= data.groupby('ssn').agg(
    customer_id =('customer_id', 'max'),
    age = ('age', lambda x: get_value_on_max_month(x, data.loc[x.index, 'month'])),
    current_occupation=('occupation', lambda x: get_value_on_max_month(x, data.loc[x.index, 'month'])),
    current_annual_income= ('annual_income', lambda x : get_value_on_max_month(x, data.loc[x.index, 'month'])),
    monthly_inhand_salary_avg =('monthly_inhand_salary', 'mean'),
    num_bank_accounts_avg = ('num_bank_accounts', 'mean'),
    num_credit_card_avg = ('num_credit_card', 'mean'),
    interest_rate_avg = ('num_of_loan', 'mean'),
    num_of_loan_avg= ('num_of_loan', 'mean'),
    delay_from_due_date_avg = ('delay_from_due_date', 'mean'),
    num_of_delayed_payment_max= ('num_of_delayed_payment', 'max'),
    current_changed_credit_limit= ('changed_credit_limit', lambda x: get_value_on_max_month(x, data.loc[x.index, 'month'])),
    num_credit_inquiries_max= ('num_credit_inquiries', 'max'),
    outstanding_debt_avg= ('outstanding_debt', 'mean'),
    current_credit_utilization_ratio= ('credit_utilization_ratio', lambda x: get_value_on_max_month(x, data.loc[x.index, 'month'])),
    current_payment_of_min_amount= ('payment_of_min_amount', lambda x: get_value_on_max_month(x, data.loc[x.index, 'month'])),
    total_amount_invested= ('amount_invested_monthly', 'sum'),
    mothly_balance_avg= ('monthly_balance', 'mean'),
    current_total_emi_per_month= ('total_emi_per_month', lambda x: get_value_on_max_month(x, data.loc[x.index, 'month'])),
    current_credit_history_age_days =('credit_history_age_days', lambda x: get_value_on_max_month (x, data.loc[x.index, 'month'])),
    credit_score= ('credit_score', lambda x: get_value_on_max_month(x, data.loc[x.index, 'month']))    
)

data_tf

data_tf['credit_score'] = data_tf['credit_score'].map({'Standard': 1, 'Poor':0, 'Good': 2})
data_tf['current_payment_of_min_amount']= data_tf['current_payment_of_min_amount'].map({'No': 0, 'Yes': 1})

data_tf

# variables created , some about current period and some from aggregations on time like average , sum , max

data_tf.describe().applymap(lambda x: "{:,.2f}".format(x)).T

nols= 2
nrows= 10

fig, ax= plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,30))

for i , col in enumerate(data_tf.select_dtypes('number').columns):
    
    idx_col = i // nrows
    idx_row = i % nrows

    #print(idx_row, idx_col, col)
    
    sns.boxplot(data_tf[col], ax= ax[idx_row, idx_col])
    sns.stripplot(data_tf[col], ax= ax[idx_row, idx_col], alpha= 0.4)
    ax[idx_row, idx_col].set_title(f'{str.title(col)}')
    

def remove_outliers(df):
    
    df= df.copy()
    
    idx_list= []
    
    for col in df.select_dtypes('number').columns.tolist():
        
        if col != 'credit_score':
            
            q1= np.nanpercentile(df[col], 25)
            q3= np.nanpercentile(df[col], 75)
            
            iqr= q3- q1
            
            lower_lim = q1 - 1.5 * iqr
            upper_lim = q3 + 1.5 * iqr
            
            outliers = df[(df[col] < lower_lim) & (df[col] > upper_lim)].index.tolist()
            idx_list.extend(outliers)   #Indices of outliers are collected in idx_list. This is useful for debugging or logging
            
            df= df[(df[col] >= lower_lim) & (df[col] <= upper_lim)]

    print(f'Total rows remaining: {df.shape[0]}')
    
    return df  # The function returns both the filtered dataframe and the list of outlier indices.

df_n_out= remove_outliers(data_tf)
df_n_out.head()

nols= 2
nrows= 10

fig, ax= plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,30))

for i , col in enumerate(df_n_out.select_dtypes('number').columns):
    
    idx_col = i // nrows
    idx_row = i % nrows

    #print(idx_row, idx_col, col)
    
    sns.boxplot(df_n_out[col], ax= ax[idx_row, idx_col])
    sns.stripplot(df_n_out[col], ax= ax[idx_row, idx_col], alpha= 0.4)
    ax[idx_row, idx_col].set_title(f'{str.title(col)}')    


# num_of_loan are negative 



df_n_out['num_of_loan_avg'].describe().T

# data_tf.to_csv('data_tf.csv')

df_n_out = df_n_out[df_n_out['num_of_loan_avg'] > 0 ]

df_n_out.shape


nols= 2
nrows= 10

fig, ax= plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,30))

for i , col in enumerate(df_n_out.select_dtypes('number').columns):
    
    idx_col = i // nrows
    idx_row = i % nrows

    #print(idx_row, idx_col, col)
    
    sns.boxplot(df_n_out[col], ax= ax[idx_row, idx_col])
    sns.stripplot(df_n_out[col], ax= ax[idx_row, idx_col], alpha= 0.4)
    ax[idx_row, idx_col].set_title(f'{str.title(col)}')   
    

# Still have outliers but keepp it

ax = sns.barplot(data= df_n_out, x= 'credit_score', y= 'monthly_inhand_salary_avg', errorbar=None)
sns.set_style('white')
plt.xticks([0,1,2], ['Poor', 'Standard', 'Good'])
labels= ax.containers
plt.bar_label(labels[0])
plt.title('Salary x Credit Score')
plt.show()

ax= sns.barplot(data=df_n_out, x= 'credit_score', y= 'age', errorbar= None)
sns.set_style('dark')
plt.xticks([0,1,2], ['Poor', 'Standard', 'Good'])
labels= ax.containers
plt.bar_label(labels[0])
plt.title('Age x Credit Score')
plt.show()


ax= sns.barplot(data=df_n_out, x= 'credit_score', y= 'current_credit_utilization_ratio', errorbar= None)
sns.set_style('dark')
plt.xticks([0,1,2], ['Poor', 'Standard', 'Good'])
labels= ax.containers
plt.bar_label(labels[0])
plt.title('Curr x Utilization x Credit Score')
plt.show()

df_conting= pd.crosstab(df_n_out['current_payment_of_min_amount'], df_n_out['credit_score'])
df_conting

from scipy.stats import chi2_contingency

chi2, p, dof, expected= chi2_contingency(df_conting)

print(f"X2 : {chi2}")
print(f"P-value: {p}")
print(f"DOF: {dof}")
print(f"Expected Freq: \n{expected}")

# null hypothesis : there is no significant association between the two variables
# reject HO , so we have association between these 2 variables (likely dependent) 
# there is significant relationship between the two variables

ax= sns.barplot(data=df_n_out, x= 'credit_score', y= 'num_of_delayed_payment_max', errorbar= None)
sns.set_style('dark')
plt.xticks([0,1,2], ['Poor', 'Standard', 'Good'])
labels= ax.containers
plt.bar_label(labels[0])
plt.title('Max Num of Delayed Payment x Credit Score')
plt.show()

ax= sns.barplot(data=df_n_out, x= 'credit_score', y= 'total_amount_invested', errorbar= None)
sns.set_style('darkgrid')
plt.xticks([0,1,2], ['Poor', 'Standard', 'Good'])
labels= ax.containers
plt.bar_label(labels[0])
plt.title('Total Invested x Credit Score')
plt.show()

ax= sns.barplot(data=df_n_out, x= 'credit_score', y= 'num_of_loan_avg', errorbar= None)
sns.set_style('darkgrid')
plt.xticks([0,1,2], ['Poor', 'Standard', 'Good'])
labels= ax.containers
plt.bar_label(labels[0])
plt.title('Num of Loan Average x Credit Score')
plt.show()

df_conting2= pd.crosstab(df_n_out['current_occupation'], df_n_out['credit_score'])
df_conting2

chi2, p, dof, expected= chi2_contingency(df_conting2)

print(f"X2 : {chi2}")
print(f"P-value: {p}")
print(f"DOF: {dof}")
print(f"Expected Freq: \n{expected}")

# There is no association between credit score and occupation

# Multicolinearity Validation

plt.figure(figsize=(15,15))
sns.heatmap(data = df_n_out.corr(), annot=True, fmt='.2f', cbar=False)

def remove_colinear_vars(df, lim:float, target:pd.Series):
    
    df= df.copy()
    
    combinations= list(itertools.combinations(df.select_dtypes('number'), 2))
    # combinations1= pd.DataFrame(combinations)
    # combinations1.to_csv('combination.csv')
    vars_to_drop= []
    
    for combination in combinations:

        df_corr= df.loc[:, combination]
        
        x= df_corr.iloc[:, 0]
        y= df_corr.iloc[:, 1]
        
        corr_xy= abs(stats.pearsonr(x,y)[0])
        corr_x_target= abs(stats.pearsonr(x, target)[0])
        corr_y_target= abs(stats.pearsonr(y, target)[0])
        
        if corr_xy >= lim and corr_x_target > corr_y_target:
            
            vars_to_drop.append(y.name)
            
        elif corr_xy >= lim and corr_x_target < corr_y_target:
            
            vars_to_drop.append(x.name)
            
    df= df.drop(vars_to_drop, axis =1)
    
    print(f'Columns removed: \n{vars_to_drop}')
    
    return df 


df_reduced = remove_colinear_vars(df_n_out, 0.7, df_n_out['credit_score'].astype(float))

df_reduced

df_reduced['credit_score'].value_counts()

# Modeling

X= df_reduced.drop(['credit_score', 'customer_id'], axis=1)
y= df_reduced['credit_score']

X_train, X_test, y_train, y_test= train_test_split(X, y, train_size=0.7, random_state= 0,
                                                   stratify= df_reduced['credit_score'])

print(f'Shape Train:  {X_train.shape}\nShape Test: {X_test.shape}')


# Separte data as num_vars and cat_vars

num_vars= X.select_dtypes(include='number').columns.tolist()
num_vars
cat_vars= X.select_dtypes(exclude='number').columns.tolist()
cat_vars

# Logistic Regression OVR

scaler= StandardScaler()

oh= OneHotEncoder()

model= LogisticRegression(multi_class= 'ovr', class_weight='balanced')

preprocessor= ColumnTransformer(transformers=[('num_prep', scaler, num_vars),
                                             ('cat_prep', oh, cat_vars)])

pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

pipe.fit(X_train, y_train)

y_pred_train = pipe.predict(X_train)
y_pred_test= pipe.predict(X_test)

print(f'{pd.crosstab(y_train, y_pred_train)}\n\n{classification_report(y_train, y_pred_train)}')

print(f'{pd.crosstab(y_test, y_pred_test)}\n\n{classification_report(y_test, y_pred_test)}')


# Random Forest

scaler= StandardScaler()
oh= OneHotEncoder()

model = RandomForestClassifier(class_weight='balanced')

preprocessor= ColumnTransformer(transformers=[('num_prep', scaler, num_vars), 
                                              ('cat_prep', oh, cat_vars)])

pipe= Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

pipe.fit(X_train, y_train)


# Pipeline
# Combines the preprocessing steps (preprocessor) and the machine learning model (model) into one pipeline.
# The steps parameter contains a list of tuples, where:
# The first tuple applies the ColumnTransformer.
# The second tuple fits the RandomForestClassifier.
# When the pipeline's fit() method is called:

# Preprocessing is applied to the training data (X_train).
# The preprocessed data is passed to the RandomForestClassifier for model training.

y_pred_train= pipe.predict(X_train)
y_pred_test= pipe.predict(X_test)

print(f'{pd.crosstab(y_train, y_pred_train)}\n\n{classification_report(y_train, y_pred_train)}')

print(f'{pd.crosstab(y_test, y_pred_test)}\n\n{classification_report(y_test, y_pred_test)}')


# Gaussian Naive Bayes

scaler= StandardScaler()
oh= OneHotEncoder()
model= GaussianNB()

preprocessor= ColumnTransformer(transformers=[('num_prep', scaler, num_vars),
                                              ('cat_prep', oh, cat_vars)])

pipe = Pipeline(steps=[('preprocessor',preprocessor),('model', model)])

pipe.fit(X_train, y_train)


y_pred_train= pipe.predict(X_train)
y_pred_test= pipe.predict(X_test)


print(f'{pd.crosstab(y_train, y_pred_train)}\n\n{classification_report(y_train, y_pred_train)}')

print(f'{pd.crosstab(y_test, y_pred_test)}\n\n{classification_report(y_test, y_pred_test)}')



# Best Model + Tunning

scaler= StandardScaler()
oh= OneHotEncoder()
model= RandomForestClassifier()

preprocessor= ColumnTransformer(transformers=[('num_prep', scaler, num_vars),
                                              ('cat_prep', oh, cat_vars)])

pipe= Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

param_grid= {
    'model__n_estimators' :[25, 50,100],
    'model__max_depth' : [5,10],
    'model__min_samples_split' : [2,5,10],
    'model__min_samples_leaf': [1,2,4],
    'model__class_weight' : ['balanced']
}

grid= GridSearchCV(pipe, param_grid=param_grid, cv=10, n_jobs=-1, scoring='accuracy', verbose=1)

grid.fit(X_train, y_train)




# Reference Documents 

import os

# Reference document path
reference_doc= r"C:\Users\rohan\CreditRisk\ref_confusionmatrix.docx"


# Function to open the Word document
def open_reference_doc():
    print(f"Opening reference document: {reference_doc}")
    os.startfile(reference_doc)  # Works on Windows

# Example: Attach this to a specific code section
if __name__ == "__main__":
    open_reference_doc()