
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
import pandas as pd
import datetime
import time
import itertools
import pickle

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

path_main_folder = '/home/antorosi/Documents/Prediction'
sys.path.append(path_main_folder)

from conso.load_shape_data import load_data_conso, get_uniformed_data_conso, change_granularity, get_x_y_prediction_conso, get_train_test_sets, normalized_dataset, select_variables


# ### Load and shape data 

# In[2]:


# Load
path_data = os.path.join(path_main_folder, 'data')
dict_data_conso = load_data_conso(path_data)

# Uniformization
data_conso_df, dict_colnames_conso = get_uniformed_data_conso(dict_data_conso)

# Granularity from 15 min to 1H
data_conso_df = change_granularity(data_conso_df, granularity="1H")

# Get x and y from prediction
x_conso, y_conso, dict_colnames_conso = get_x_y_prediction_conso(data_conso_df, dict_colnames_conso, lags=[24,48])


# ### Cross-validation parameters 

# In[3]:


# folder to store results
path_out = os.path.join(path_main_folder, 'out', 'benchmark_cmca_rf')
if not os.path.exists(path_out):
    os.mkdir(path_out)


# In[4]:


# variables used for input
selected_variables = ['conso', 'calendar', 'meteo']
gen_name = 'cmca'


# In[5]:


# Test periods for each K step of the cross-validation
cv_periods = {}
cv_periods['period_1'] = (datetime.datetime(year=2013, month=1, day=1), datetime.datetime(year=2013, month=12, day=31))
cv_periods['period_2'] = (datetime.datetime(year=2014, month=1, day=1), datetime.datetime(year=2014, month=12, day=31))
cv_periods['period_3'] = (datetime.datetime(year=2015, month=1, day=1), datetime.datetime(year=2015, month=12, day=31))
cv_periods['period_4'] = (datetime.datetime(year=2016, month=1, day=1), datetime.datetime(year=2016, month=12, day=31))
cv_periods['period_5'] = (datetime.datetime(year=2017, month=1, day=1), datetime.datetime(year=2017, month=12, day=31))


# In[6]:


# Getting each datasets
dict_datasets = {}
for key, date_period in cv_periods.items():
    x_conso_selected_var = select_variables(x_conso, dict_colnames_conso, selected_variables)
    dataset, dict_ds = get_train_test_sets(x_conso_selected_var, y_conso, date_period[0], date_period[1])
    dataset = normalized_dataset(dataset, dict_colnames_conso)
    
    dict_datasets[key] = {'dataset': dataset, 'dict_ds': dict_ds}


# ### Dataset

# In[36]:


date_period = [datetime.datetime(2013,1,1), datetime.datetime(2017,12,31)]

x_conso_selected_var = select_variables(x_conso, dict_colnames_conso, selected_variables)
dataset, dict_ds = get_train_test_sets(x_conso_selected_var, y_conso, date_period[0], date_period[1])
dataset = normalized_dataset(dataset, dict_colnames_conso)

dataset = dataset['test']
dict_ds=dict_ds['test']

x = dataset['x']
y = dataset['y']


# In[37]:


train_indices = list()
test_indices = list()

for year in [2013,2014,2015,2016,2017]:
    mask = dict_ds.dt.year == year
    test_indices.append(np.where(mask)[0])
    train_indices.append(np.where(np.invert(mask))[0])

custom_cv = zip(train_indices, test_indices)


# In[38]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 400, num = 8)]
# Number of features to consider at every split
max_features = ['sqrt', 1/3]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 6, 8, 10, 15]
# Method of selecting samples for training each tree
bootstrap = [True]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[ ]:


rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(scoring='neg_mean_squared_error', estimator = rf, param_distributions = random_grid, n_iter = 200, cv = custom_cv, verbose=2, random_state=44, n_jobs = -1)

# Fit the random search model
rf_random.fit(X=x, y=np.ravel(y))


# In[ ]:


with open(os.path.join(path_out,'results.pickle'), 'wb') as f:
    pickle.dump(rf_random.cv_results_,f)

