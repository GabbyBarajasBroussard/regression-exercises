#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import env
from sklearn.model_selection import train_test_split
import sklearn.preprocessing


# In[ ]:


# creating a connection to connect to the Codeup Student Database
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# In[ ]:


def get_wrangle_telco_data():
    '''This function will connect to the Codeup Student Database. It will then cache a local copy to the computer to use for later
        in the form of a CSV file. If you want to reproduce the results, you will need your own env.py file and database credentials.'''
    filename = "wrangle_telco.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('''
    SELECT customer_id, monthly_charges, tenure, total_charges
    FROM customers
    WHERE contract_type_id = 3''' , get_connection('telco_churn'))
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)
        # Return the dataframe to the calling code
        return df


# In[ ]:
def wrangle_telco(df):
    df= get_wrangle_telco_data()
    del df['Unnamed: 0']
    # Changes total_charges to a float numeric variable by removing spaces from string
    df.total_charges = df.total_charges.str.replace(' ', '0').astype(float)
    # Replaces NaN values with 0 for new customers with no total_charges
    df["total_charges"].fillna(0, inplace = True) 
    return df

