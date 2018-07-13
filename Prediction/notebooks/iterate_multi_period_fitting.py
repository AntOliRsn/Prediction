
# coding: utf-8

# In[45]:


import os
import sys
import numpy as np
import pandas as pd
import datetime
import time
import itertools
import pickle
import json

from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

path_main_folder = '/home/antorosi/Documents/Prediction'
sys.path.append(path_main_folder)

from conso.load_shape_data import load_data_conso, get_uniformed_data_conso, change_granularity, get_x_y_prediction_conso, get_train_test_sets, normalized_dataset, select_variables, add_atypical_events_to_dict_data_conso
from models.feedforward_NN import FeedForward
from aed.atypical_event import AtypicalEventsList, AtypicalEvent, dataframe_daily_to_ael
from aed.detection import get_next_hd_events, sort_atypical_events, get_residuals, get_atypical_score, get_atypical_ds, prediction_conso_plot, aed_conso_plot
from aed.multi_period import get_prediction_results, get_aed_results, get_complete_df


# ### Load and shape data 

# In[46]:


# Load
path_data = os.path.join(path_main_folder, 'data')
dict_data_conso = load_data_conso(path_data)


# ### Folder to store results  

# In[47]:


# folder to store results
path_out = os.path.join(path_main_folder, 'out', 'iterate_FFNN_2')
if not os.path.exists(path_out):
    os.mkdir(path_out)


# In[48]:


path_data = os.path.join("/local/home/antorosi/Documents/Prediction/data")
with open(os.path.join(path_data, 'ae_reference_list_wwe_2013-2017' + '.pickle'), 'rb') as f:
    ael_reference= pickle.load(f)


# ### Iterative fitting parameters 

# In[49]:


# iterative parameters
nb_iter = 20
nb_ae = 50

# input parameters
selected_variables = ['calendar', 'conso','meteo', 'atypical_events']
gen_name = 'cmcaae'

# model parameters
nb_hidden_units = 171
nb_hidden_layers = 2
dropout = 0.1
training_epochs=400
batch_size=100


# In[50]:


# atypical event list initialisation
ael = AtypicalEventsList()


# In[51]:


# Test periods for each K step of the cross-validation
cv_periods = {}
cv_periods['period_1'] = (datetime.datetime(year=2013, month=1, day=1), datetime.datetime(year=2013, month=12, day=31))
cv_periods['period_2'] = (datetime.datetime(year=2014, month=1, day=1), datetime.datetime(year=2014, month=12, day=31))
cv_periods['period_3'] = (datetime.datetime(year=2015, month=1, day=1), datetime.datetime(year=2015, month=12, day=31))
cv_periods['period_4'] = (datetime.datetime(year=2016, month=1, day=1), datetime.datetime(year=2016, month=12, day=31))
cv_periods['period_5'] = (datetime.datetime(year=2017, month=1, day=1), datetime.datetime(year=2017, month=12, day=31))


# ### Main loop

# In[44]:


for iteration in range(nb_iter):
    ### Create new folder to store iteration results
    path_out_iter = os.path.join(path_out, 'iter_'+str(iteration))
    if not os.path.exists(path_out_iter):
        os.mkdir(path_out_iter)
    
    ### UPDATE DATA WITH NEW ATYPICAL EVENTS ###
    dict_data_conso = add_atypical_events_to_dict_data_conso(dict_data_conso, ael)

    # Uniformization
    data_conso_df, dict_colnames_conso = get_uniformed_data_conso(dict_data_conso)

    # Granularity from 15 min to 1H
    data_conso_df = change_granularity(data_conso_df, granularity="1H")

    # Get x and y from prediction
    x_conso, y_conso, dict_colnames_conso = get_x_y_prediction_conso(data_conso_df, dict_colnames_conso, lag=24)

    # Getting each datasets
    dict_datasets = {}
    for key, date_period in cv_periods.items():
        x_conso_selected_var = select_variables(x_conso, dict_colnames_conso, selected_variables)
        dataset, dict_ds = get_train_test_sets(x_conso_selected_var, y_conso, date_period[0], date_period[1])
        dataset = normalized_dataset(dataset, dict_colnames_conso)

        dict_datasets[key] = {'dataset': dataset, 'dict_ds': dict_ds}
    
    # Deleting useless info
    del x_conso, y_conso
    
    ### Getting FFNN input dim ###
    input_dim = dict_datasets['period_1']['dataset']['train']['x'].shape[1]

    ### Prepare results wrap up ###
    results_df = pd.DataFrame(columns=['name', 'layer_dims','dropout_rates','batchsize',
                                               'best_iter', 'train_mse',
                                               'train_mae', 'train_mape',
                                               'test_mse', 'test_mae',
                                               'test_mape'])
    path_results = path_out_iter
    
    ### TRAINNING ###
    
    idx = 0

    for name_period, el in dict_datasets.items():
        dataset = el['dataset']

        print('========================= Iteration {} Model {} ========================='.format(iteration+1,idx+1))

        # Prepare model characteristics
        name_model = '{}_FFNN_l{}*{}_d{}*{}_{}_norm'.format(name_period,nb_hidden_units, nb_hidden_layers,
                                                            dropout, nb_hidden_layers, 
                                                            gen_name)

        # Compile model
        model = FeedForward(name=name_model, output=path_out_iter, input_dim=input_dim, output_dim=1, 
                   l_dims=[nb_hidden_units]*nb_hidden_layers, dropout_rates=[dropout]*nb_hidden_layers,
                   loss = 'mean_squared_error', metrics = ['mape', 'mae'])

        # Prepare callbacks
        callbacks = []
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100,
                                                           verbose=0, mode='auto')

        model_checkpoint = ModelCheckpoint(os.path.join(path_out_iter, name_model, 'models', 'model-best.hdf5'),
                                                   monitor='val_loss',
                                                   verbose=0, save_best_only=True, save_weights_only=False,
                                                   mode='auto', period=1)

        tensorboard_model = TensorBoard(log_dir=os.path.join(path_out_iter, name_model, 'results', 'logs', time.strftime('%Y-%m-%d_%H:%M', time.localtime(time.time()))))
        tensorboard_summary = TensorBoard(log_dir=os.path.join(path_out_iter, 'logs', name_model))


        callbacks.append(early_stop)
        callbacks.append(model_checkpoint)
        callbacks.append(tensorboard_model)
        callbacks.append(tensorboard_summary)

        # Train model
        model.main_train(dataset, training_epochs=training_epochs, batch_size=batch_size, callbacks=callbacks)

        # Get result and put it in results
        _, result = model.analyze_history(dataset)

        results_df = results_df.append(result, ignore_index=True)
        results_df.to_csv(os.path.join(path_results, 'cv_results.csv'), sep=';')
        
        # Reset graph
        K.clear_session()
        import tensorflow as tf
        tf.reset_default_graph()

        idx += 1

    ### Atypical Event Detection ###
    path_models_folder = path_out_iter
    mode = 2
    type_model = 'keras'
    
    # get prediction
    prediction_results = get_prediction_results(path_models_folder, dict_datasets, mode, type_model)
    
    # get atypical events info
    threshold = 0.95
    aed_results, ael_full_model = get_aed_results(prediction_results, threshold)
    
    # get complete dataframe
    atypical_full_df, prediction_full_df = get_complete_df(prediction_results, aed_results)
    
    # sort events
    atypical_new_df = sort_atypical_events(atypical_full_df, ael_reference)
    
    # get ael
    selected_ae_iter = atypical_new_df[:nb_ae]
    ael_iter = dataframe_daily_to_ael(selected_ae_iter)
    
    # merge ael
    ael = ael.get_union(ael_iter)
    
    ### Save results ###

    # plot 
    # Not working with the current method
    path_plot = path_results
    #name_plot = 'full_period_mode{}_t{}_iter{}.html'.format(mode, threshold, iteration)
    #aed_conso_plot(data_conso_df, atypical_full_df, prediction_full_df, dict_colnames_conso, path_plot, name_plot)
    
    # data
    with open(os.path.join(path_results, 'atypical_full_df.pickle'),'wb') as f:
        pickle.dump(atypical_full_df,f)
    with open(os.path.join(path_results, 'prediction_full_df.pickle'),'wb') as f:
        pickle.dump(prediction_full_df,f)
    with open(os.path.join(path_results, 'ael.pickle'),'wb') as f:
        pickle.dump(ael,f)
    with open(os.path.join(path_results, 'ael_iter.pickle'),'wb') as f:
        pickle.dump(ael_iter,f)
    
    ael.get_events_list().to_csv(os.path.join(path_results, 'ael.csv'), sep=';')
    

