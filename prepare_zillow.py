#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os
import env
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

def get_zillow_data():
    '''This function will connect to the Codeup Student Database. It will then cache a local copy to the computer to use for later
        in the form of a CSV file. If you want to reproduce the results, you will need your own env.py file and database credentials.'''
    filename = "zillow.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('''
            SELECT * FROM properties_2017
            JOIN predictions_2017 ON properties_2017.parcelid = predictions_2017.parcelid
            JOIN airconditioningtype ON properties_2017.airconditioningtypeid = airconditioningtype.airconditioningtypeid
            JOIN heatingorsystemtype ON properties_2017.heatingorsystemtypeid = heatingorsystemtype.heatingorsystemtypeid
            JOIN architecturalstyletype ON properties_2017.architecturalstyletypeid = architecturalstyletype.architecturalstyletypeid
            JOIN propertylandusetype ON properties_2017.propertylandusetypeid = propertylandusetype.propertylandusetypeid
            JOIN typeconstructiontype ON properties_2017.typeconstructiontypeid = typeconstructiontype.typeconstructiontypeid
            WHERE properties_2017.propertylandusetypeid = '261' OR '262' OR '263' OR '264' OR '268' OR '273' OR '274' OR '275' OR '276' OR '279'
            AND transactiondate BETWEEN '2017-05-01' AND '2017-06-30';
            ''' , get_connection('zillow'))
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)
        # Return the dataframe to the calling code
        return df


# In[2]:


def clean_zillow():
    df= get_zillow_data()
    #rename columns to make it easiet to call later
    df= df.rename(columns={"parcelid": "parcel_id", "bedroomcnt": "bedroom_count","bathroomcnt": "bathroom_count",
                      "calculatedfinishedsquarefeet": "square_feet", "airconditioningdesc": "airconditioning",
                      "heatingorsystemdesc": "heating", "architecturalstyledesc": "architectural_style",
                      "propertylandusedesc": "property_type", "typeconstructiondesc": "construction_type",
                      "poolcnt": "pool_count", "roomcnt": "room_count", "taxvaluedollarcnt": "tax_value_dollar_count",
                      "structuretaxvaluedollarcnt": "structure_tax_value_dollar_count","landtaxvaluedollarcnt": "land_tax_value_dollar_count", 
                      "taxmount": "tax_amount","garagecarcnt": "car_size_garage"
                      })
    # drop columns with more than 50% (58 count) nulls and duplicate/similar columns
    df= df.drop(columns=['garagetotalsqft','fullbathcnt','calculatedbathnbr', 'finishedsquarefeet12',
                     'heatingorsystemtypeid','id','propertylandusetypeid','airconditioningtypeid',
                     'architecturalstyletypeid','typeconstructiontypeid.1','parcelid.1', 'parcelid.1', 
                     'id.1','Unnamed: 0','basementsqft','buildingclasstypeid', 'buildingqualitytypeid', 
                     'decktypeid', 'finishedfloor1squarefeet','finishedsquarefeet13', 'finishedsquarefeet15',
                     'finishedsquarefeet50','finishedsquarefeet6','fireplacecnt','hashottuborspa', 
                     'lotsizesquarefeet', 'poolsizesum', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 
                     'propertyzoningdesc', 'regionidneighborhood', 'storytypeid', 'threequarterbathnbr', 
                     'unitcnt','yardbuildingsqft17','yardbuildingsqft26','fireplaceflag', 
                     'airconditioningtypeid.1','heatingorsystemtypeid.1','architecturalstyletypeid.1', 
                     'propertylandusetypeid.1', 'taxdelinquencyflag', 'taxdelinquencyyear','typeconstructiontypeid'])

    #handle any missing values
    df['pool_count'].fillna(df['pool_count'].mode()[0], inplace=True)
    # return the clean dataframe
    return df


# In[3]:


#combining my split, train, test data and my clean data into one dataframe
def prep_zillow_data():
    df= clean_zillow()
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123) 
    return train, validate, test
def prep_split_zillow_data():
    df=clean_zillow()
    X_train = train.drop(columns='tax_value_dollar_count')
    X_validate = validate.drop(columns='tax_value_dollar_count')
    X_test = test.drop(columns='tax_value_dollar_count')

    y_train = train['tax_value_dollar_count']
    y_validate = validate['tax_value_dollar_count']
    y_test = test['tax_value_dollar_count']
    return X_train, X_validate, X_test, y_train, y_validate, y_test


# In[ ]:




