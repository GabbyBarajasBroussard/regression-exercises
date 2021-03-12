
import pandas as pd
import numpy as np
import os
import env
from sklearn.model_selection import train_test_split
import sklearn.preprocessing


# creating a connection to connect to the Codeup Student Database
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_telco_data():
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

def wrangle_telco():
    '''This takes in the data from the get_telco_data, cleans the data, and splits into train, validate, and test.'''
    df= get_telco_data()
    # Changes total_charges to a float numeric variable by removing spaces from string
    df.total_charges = df.total_charges.str.replace(' ', '0').astype(float)
    # Replaces NaN values with 0 for new customers with no total_charges
    df["total_charges"].fillna(0, inplace = True) 
    df= df.drop(columns=['Unnamed: 0','customer_id'])
    train_and_validate, test = train_test_split(df, test_size=.15, random_state=123)
    train, validate = train_test_split(train_and_validate, test_size=.15, random_state=123)
    #return train, validate, test
    return train, validate, test

def add_scaled_columns(train, validate, test, scaler, columns_to_scale):
    '''This function takes in a train, validate and test as well as columns needed to be scaled. Then a scaled train, validate, test is returned'''
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler.fit(train[columns_to_scale])

    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    return train, validate, test

def scaled_telco(train, validate, test):
    '''This function takes in a train, validate and test and returns a min, max scaled version of each.'''
    train, validate, test = add_scaled_columns(
    train,
    validate,
    test,
    scaler=sklearn.preprocessing.MinMaxScaler(),
    columns_to_scale=['monthly_charges', 'total_charges', 'tenure'],
    )
    return train, validate, test

def inverse_scaled_columns(train, validate, test, scaler, columns_to_scale, columns_to_inverse):
    '''This function takes in a train, validate and test as well as columns needed to be scaled and inversed. Then a scaled and inversed train, validate, test is returned'''
    new_column_names = [c + '_inverse' for c in columns_to_inverse]
    scaler.fit(train[columns_to_scale])

    train = pd.concat([
        train,
        pd.DataFrame(scaler.inverse_transform(train[columns_to_inverse]), columns=new_column_names, index=train.index),
    ], axis=1)
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.inverse_transform(validate[columns_to_inverse]), columns=new_column_names, index=validate.index),
    ], axis=1)
    test = pd.concat([
        test,
        pd.DataFrame(scaler.inverse_transform(test[columns_to_inverse]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    return train, validate, test