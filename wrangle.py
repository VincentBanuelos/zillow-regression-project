import os
import pandas as pd
import numpy as np

from env import user, password, host

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")



####################################################################################################ACQUIRE##############################################################################################################################

def get_zillow():

    if os.path.isfile('zillow.csv'):
        return pd.read_csv('zillow.csv', index_col=0)
    
    else:

        url = f"mysql+pymysql://{user}:{password}@{host}/zillow"
        
        query = '''
                SELECT parcelid, bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet as sqft, fips as county, fireplacecnt,
                       garagecarcnt, lotsizesquarefeet, yearbuilt, taxdelinquencyflag, poolcnt,
                       transactiondate, propertylandusedesc, taxvaluedollarcnt as tax_value, regionidzip as zip,latitude,longitude
                FROM properties_2017
                JOIN predictions_2017
                USING (parcelid)
                JOIN propertylandusetype
                USING (propertylandusetypeid)
                HAVING propertylandusedesc = 'Single Family Residential'
                '''

        df = pd.read_sql(query, url)

        df.to_csv('zillow.csv')

    return df

####################################################################################################PREPARE##############################################################################################################################
def decades(x):
    if x > 1899 and x < 1910:
        return 1900
    if x > 1909 and x < 1920:
        return 1910
    if x > 1919 and x < 1930:
        return 1920
    if x > 1929 and x < 1940:
        return 1930
    if x > 1939 and x < 1950:
        return 1940
    if x > 1949 and x < 1960:
        return 1950
    if x > 1959 and x < 1970:
        return 1960
    if x > 1969 and x < 1980:
        return 1970
    if x > 1979 and x < 1990:
        return 1980
    if x > 1989 and x < 2000:
        return 1990
    if x > 1999 and x < 2010:
        return 2000
    if x > 2009:
        return 2010

def prep_zillow(df):
    '''
    Accepts the zillow dataframe and prepares it for modeling.

    '''

    # filtering for only transactions in 2017
    df = df[df.transactiondate < '2018-01-01']

    # Dropping nulls in sqft
    df = df[df['sqft'].notna()]

    # create column with fips value converted from an integer to the county name string
    df['county'] = df.county.map({6037 : 'los_angeles', 6059 : 'orange', 6111 : 'ventura'})

    # dropping nulls in yearbuilt
    df = df[df['yearbuilt'].notna()]

    # dropping nulls in zip
    df = df[df['zip'].notna()]

    # Creating decade column to split yearbuilt in decades.

    df['decade'] = df['yearbuilt'].apply(decades)
    df.decade = df.decade.astype(object)

    # convert poolcnt nulls to 0's
    df.poolcnt = df.poolcnt.fillna(0)

    # convert fireplace count nulls to 0
    df.fireplacecnt = df.fireplacecnt.fillna(0)

    # garage null values to 0
    df.garagecarcnt = df.garagecarcnt.fillna(0)

    # Dropping nulls in lotsizesquarefeet
    df = df[df['lotsizesquarefeet'].notna()]

    # Dropping rows where the porperties are Tax deliquent
    df.drop(df[df['taxdelinquencyflag'] == 'Y'].index, inplace = True)
    
    # Dropping propertylandusedesc and taxdelinquencyflag since all rows will have the same values
    df.drop(columns=['propertylandusedesc','taxdelinquencyflag'], inplace=True)

    # rename columns for clarity
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms', 'calculatedfinishedsquarefeet':'sqft','taxvaluedollarcnt':'tax_value', 'lotsizesquarefeet':'lotsize'})

    # one-hot encode county
    dummies = pd.get_dummies(df['county'],drop_first=False,dtype=float)
    df = pd.concat([df, dummies], axis=1)

    return df

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

####################################################################################################WRANGLE##############################################################################################################################


def wrangle_zillow():

    df = get_zillow()
    df = prep_zillow(df)
    col_list = ['bathrooms', 'bedrooms', 'sqft', 'lotsize', 'yearbuilt', 'tax_value']
    df = remove_outliers(df, 1.5, col_list)
                        
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    return train, validate, test




####################################################################################################SCALE DATA##############################################################################################################################

def xy_tvt_split(train, validate, test, target):
    """
    this function takes in a dataframe and splits it into 3 samples,
    a test, which is 20% of the entire dataframe,
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe.
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    """
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=['parcelid','county','decade','transactiondate','tax_value'])
    y_train = train[target]

    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=['parcelid','county','decade','transactiondate','tax_value'])
    y_validate = validate[target]

    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=['parcelid','county','decade','transactiondate','tax_value'])
    y_test = test[target]

    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)

    return X_train, y_train, X_validate, y_validate, X_test, y_test



def min_max_scale(X_train, X_validate, X_test):
    """
    this function takes in 3 dataframes with the same columns,
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler.
    it returns 3 dataframes with the same column names and scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = MinMaxScaler(copy=True).fit(X_train)

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns.values).set_index([X_train.index.values])
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), columns = X_validate.columns.values).set_index([X_validate.index.values])
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns.values).set_index([X_test.index.values])

    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    

    return X_train_scaled, X_validate_scaled, X_test_scaled


