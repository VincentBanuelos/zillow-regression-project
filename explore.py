import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydataset
import seaborn as sns
sns.set()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

from sklearn.model_selection import train_test_split
from scipy import stats as stats
from itertools import combinations


import warnings
warnings.filterwarnings("ignore")

import wrangle as wr

def plot_categorical_and_continuous_vars(df):
    columns = ['bedrooms', 'bathrooms', 'sqft',]
    for col in columns:
            plt.figure(figsize=(10, 7))
            sns.histplot(data = df, x=col, hue= 'county', element="step")
            
            plt.show()
    return

def zillow_lmplot(df):
    """
    Takes in the train dataframe for the train dataset and returns an lmplot
    
    """
    plt.figure(figsize=(12,8))
    sns.lmplot(x='sqft', y='tax_value',data = df.sample(1000), hue = 'county', size = 8,col = 'county')
    plt.show()
    return

def county_value(df):
    '''
    Takes in county and tax_value and returns a histogram plot and a boxplot comparing the two
    '''
    plt.figure(figsize=(12,8))
    sns.histplot(data=df, x='tax_value', alpha=.8, hue='county',hue_order=['ventura', 'orange', 'los_angeles'],element="step")
    plt.title('Orange County requires the most green')
    plt.show()

    print()

    plt.figure(figsize=(12,8))
    sns.boxplot(x='county', y='tax_value', data=df)
    plt.title('Orange County requires the most green')
    plt.show()
    return

def yb_graph(df):
    '''
    shows the conparison for decade and tax_value
    '''
    plt.figure(figsize=(12,8))
    sns.histplot(data=df, x='tax_value', alpha=.8, hue='county',hue_order=['ventura', 'orange', 'los_angeles'],element="step")
    plt.title('L.A should be called LAbuela')
    plt.xticks([1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010], ['1910', '1920', '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010'])
    plt.show()

    print()

    plt.figure(figsize = (12, 8))
    ax = sns.barplot(x='decade', y='tax_value', data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    tax_value_avg = df.tax_value.mean()
    plt.axhline(tax_value_avg, label='Tax Value Average')
    plt.legend()
    plt.xlabel('Decades')
    plt.ylabel('Tax Value')
    plt.title("The 70's and newer show a not so groovy price")
    plt.show()
    return

def bb_graph(df):
    '''
    shows how bedrooms and bathrooms relate to tax_value
    '''
    plt.figure(figsize=(12,8))
    sns.boxplot(x='bedrooms', y='tax_value', data=df)
    plt.title('Tax Value Increase w/ the Number of Bedrooms')
    plt.show()
    
    print()
    
    plt.figure(figsize=(12,8))
    sns.boxplot(x='bathrooms', y='tax_value', data=df)
    plt.title('Tax Value Increase w/ the Number of Bathrooms')
    plt.show()



def sqft_test(df):
    '''
    Runs statistical testing to see if homes below or above median SQFT are more expenesive.
    '''
    # Create the samples
    sqft_above_md = df[df.sqft > df.sqft.median()].tax_value
    sqft_below_md = df[df.sqft < df.sqft.median()].tax_value

    # Set alpha
    alpha = 0.05

    # Check for equal variances
    s, pval = stats.levene(sqft_above_md, sqft_below_md)

    # Run the two-sample, one-tail T-test.
    # Use the results from checking for equal variances to set equal_var
    t, p = stats.ttest_ind(sqft_above_md, sqft_above_md, equal_var=(pval >= alpha))

    # Evaluate results based on the t-statistic and the p-value
    if p/2 < alpha and t > 0:
        print('''Reject the Null Hypothesis.
        
Properties that are ABOVE THE SQFT are MORE expensive than those BELOW MEDIAN SQFT.''')
    else:
        print('''Fail to reject the Null Hypothesis.
        
Properties that are ABOVE THE SQFT are LESS expensive than those BELOW MEDIAN SQFT.''')



def county_test(df):
    '''
    Runs statistical test to see if L.A is pricier than Orange County and Ventura combined.
    '''

    la_homes = df[df.county == 'Los Angeles'].tax_value
    vo_homes = df[(df.county == 'Orange')|(df.county == 'Ventura') ].tax_value

    # Set alpha
    alpha = 0.05

    # Check for equal variances
    s, pval = stats.levene(la_homes, vo_homes)

    # Use the results from checking for equal variances to set equal_var
    t, p = stats.ttest_ind(la_homes, vo_homes, equal_var=(pval >= alpha))

    # Evaluate results based on the t-statistic and the p-value
    if p/2 < alpha and t < 0:
        print('''Reject the Null Hypothesis.
        
Homes in Los Angeles are LESS expensive than those in either Orange County and Ventura.''')
    else:
        print('''Fail to reject the Null Hypothesis.
        
Homes in Los Angeles are MORE expensive than those in either Orange County and Ventura.''')


def yb_test(df):
    ''' 
    Runs statistical test for yearbuilt column, seeing which houses are more expensive, pre-1972 or post-1972.
    '''
    older_1972 = df[df.yearbuilt <= 1972].tax_value
    newer_1972 = df[df.yearbuilt > 1972].tax_value

    # Set alpha
    alpha = 0.05

    # Check for equal variances
    s, pval = stats.levene(older_1972, newer_1972)

    # Use the results from checking for equal variances to set equal_var
    t, p = stats.ttest_ind(older_1972, newer_1972, equal_var=(pval >= alpha))

    # Evaluate results based on the t-statistic and the p-value
    if p/2 < alpha and t > 0:
        print('''Reject the Null Hypothesis.
        
Homes built before 1972 are MORE expensive than those built afterwards.''')
    else:
        print('''Fail to reject the Null Hypothesis.
        
Homes built before or in 1972 are LESS expensive than those built afterwards.''')


def bb_test(df):
    ''' 
    Runs statitical testing for bed and bathroom comparison.
    '''
    baths_above_md = df[(df.bathrooms > df.bathrooms.median())&(df.bedrooms < df.bedrooms.median())].tax_value
    baths_below_md = df[(df.bathrooms < df.bathrooms.median())&(df.bedrooms > df.bedrooms.median())].tax_value

    # Set alpha
    alpha = 0.05

    # Check for equal variances
    s, pval = stats.levene(baths_above_md, baths_below_md)

    # Use the results from checking for equal variances to set equal_var
    t, p = stats.ttest_ind(baths_above_md, baths_below_md, equal_var=(pval >= alpha))

    # Evaluate results based on the t-statistic and the p-value
    if p/2 < alpha and t > 0:
        print('''Reject the Null Hypothesis.
        
Homes with above the median amonut of bathrooms and below the median amount of bedrooms are more expensive then the opposite.''')
    else:
        print('''Fail to reject the Null Hypothesis.
        
Homes with above the median amount of bathrooms and below the median amount of bedrooms are cheaper then the opposite.''')