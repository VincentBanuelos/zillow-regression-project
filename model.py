# standard ds imports
import numpy as np
import pandas as pd
import wrangle as wr

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# for modeling and evaluation
import wrangle
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

train, validate, test = wr.wrangle_zillow()

def scale_data(train, validate, test, target):
    '''
    
    '''
    X_train, y_train, X_validate, y_validate, X_test, y_test = wr.xy_tvt_split(train, validate, test, target)
    
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)

    X_train_scaled, X_validate_scaled, X_test_scaled = wr.min_max_scale(X_train, X_validate, X_test)

    mvp_X_train_scaled = X_train_scaled.drop(columns=['fireplacecnt','garagecarcnt', 'lotsize', 'yearbuilt', 'poolcnt', 'zip', 'latitude', 'longitude', 'los_angeles','orange', 'ventura'])
    mvp_X_validate_scaled = X_validate_scaled.drop(columns=['fireplacecnt','garagecarcnt', 'lotsize', 'yearbuilt', 'poolcnt', 'zip', 'latitude', 'longitude', 'los_angeles','orange', 'ventura'])
    mvp_X_test_scaled = X_test_scaled.drop(columns=['fireplacecnt','garagecarcnt', 'lotsize', 'yearbuilt', 'poolcnt', 'zip', 'latitude', 'longitude', 'los_angeles','orange', 'ventura'])


    X_train_scaled = X_train_scaled.drop(columns=['los_angeles','orange','ventura','zip','fireplacecnt','garagecarcnt', 'bathrooms'])
    X_validate_scaled = X_validate_scaled.drop(columns=['los_angeles','orange','ventura','zip','fireplacecnt','garagecarcnt', 'bathrooms'])
    X_test_scaled = X_test_scaled.drop(columns=['los_angeles','orange','ventura','zip','fireplacecnt','garagecarcnt', 'bathrooms'])

    return mvp_X_train_scaled, mvp_X_validate_scaled,mvp_X_test_scaled,  X_train, y_train, X_validate, y_validate, X_test, y_test, X_train_scaled, X_validate_scaled, X_test_scaled


def corr_graph():
    train_corr = train.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(train_corr, cmap='Blues', annot = True, mask= np.triu(train_corr), linewidth=.5)
    plt.show()


def run_da_stuff(X_train_scaled,y_train,X_validate_scaled,y_validate):
    pred_mean = y_train.tax_value.mean()
    y_train['pred_mean'] = pred_mean
    y_validate['pred_mean'] = pred_mean
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_mean, squared=False)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.pred_mean, squared=False)

    # save the results
    metric_df = pd.DataFrame(data=[{
        'model': 'baseline_mean',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.tax_value, y_train.pred_mean),
        'rmse_validate': rmse_validate,
        'r2_validate': explained_variance_score(y_validate.tax_value, y_validate.pred_mean)
        }])

    #Linear Regression model
    # run the model
    lm = LinearRegression(normalize=True)
    lm.fit(X_train_scaled, y_train.tax_value)
    y_train['pred_lm'] = lm.predict(X_train_scaled)
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_lm)**(1/2)
    y_validate['pred_lm'] = lm.predict(X_validate_scaled)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.pred_lm)**(1/2)

    # save the results
    metric_df = metric_df.append({
        'model': 'Linear Regression',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.tax_value, y_train.pred_lm),
        'rmse_validate': rmse_validate,
        'r2_validate': explained_variance_score(y_validate.tax_value, y_validate.pred_lm)}, ignore_index=True)


    # LassoLars Model
    lars = LassoLars(alpha=4)
    lars.fit(X_train_scaled, y_train.tax_value)
    y_train['pred_lars'] = lars.predict(X_train_scaled)
    rmse_train = mean_squared_error(y_train.tax_value, y_train.pred_lars, squared=False)
    y_validate['pred_lars'] = lars.predict(X_validate_scaled)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.pred_lars, squared=False)

    # save the results
    metric_df = metric_df.append({
        'model': 'LarsLasso, alpha 4',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.tax_value, y_train.pred_lars),
        'rmse_validate': rmse_validate,
        'r2_validate': explained_variance_score(y_validate.tax_value, y_validate.pred_lars)}, ignore_index=True)

    # create the model object
    glm = TweedieRegressor(power=0, alpha=0)
    glm.fit(X_train_scaled, y_train.tax_value)
    y_train['glm_pred'] = glm.predict(X_train_scaled)
    rmse_train = mean_squared_error(y_train.tax_value, y_train.glm_pred)**(1/2)
    y_validate['glm_pred'] = glm.predict(X_validate_scaled)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.glm_pred)**(1/2)


    # save the results
    metric_df = metric_df.append({
        'model': 'Tweedie Regressor',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.tax_value, y_train.glm_pred),
        'rmse_validate': rmse_validate,
        'r2_validate': explained_variance_score(y_validate.tax_value, y_validate.glm_pred)}, ignore_index=True)

    return metric_df

def test_tester(X_train_scaled,y_train,X_validate_scaled,y_validate,X_test_scaled,y_test):
    ''' 
    This function takes in the X and y objects and then runs and returns a DataFrame of
    results for the Quadradic Linear Regression Model. 
    '''

   # create the model object
    glm = TweedieRegressor(power=2, alpha=0) # changed power to 0 since we normalized the target
    glm.fit(X_train_scaled, y_train.tax_value)
    y_test['glm_pred'] = glm.predict(X_test_scaled)
    rmse_test = mean_squared_error(y_test.tax_value, y_test.glm_pred)**(1/2)


    # save the results
    test_metrics = pd.DataFrame({'test': 
                               {'rmse': rmse_test, 
                                'r2': explained_variance_score(y_test.tax_value, y_test.glm_pred)}
                          })
    
    return test_metrics.T