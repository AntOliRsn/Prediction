
# coding: utf-8

# ### Comments
# Modifications of the first benchmark:
# - Evolution of the layers size according to the dropout levels

# ### Import libraries

# In[20]:


import os
import sys
import numpy as np
import pandas as pd
import datetime
import time
import itertools

from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

path_main_folder = '/home/antorosi/Documents/Prediction'
#path_main_folder = os.getcwd()
sys.path.append(path_main_folder)

from conso.load_shape_data import load_data_conso, get_uniformed_data_conso, change_granularity, get_x_y_prediction_conso, get_train_test_sets, normalized_dataset, select_variables
from models.feedforward_NN import FeedForward


# ### Load and shape data

# In[21]:


# Load
path_data = os.path.join(path_main_folder, 'data')


# In[22]:


dict_data_conso = load_data_conso(path_data)


# In[23]:


# Uniformization
data_conso_df, dict_colnames_conso = get_uniformed_data_conso(dict_data_conso)


# In[24]:


# Granularity from 15 min to 1H
data_conso_df = change_granularity(data_conso_df, granularity="1H")


# In[25]:


# Get x and y from prediction
lags = [24,48,168]
x_conso, y_conso, dict_colnames_conso = get_x_y_prediction_conso(data_conso_df, dict_colnames_conso, lags=lags)


# ### Benchmark parameters 

# In[26]:


path_out = os.path.join(path_main_folder, 'out', 'benchmark_4')
if not os.path.exists(path_out):
    os.mkdir(path_out)


# In[27]:


dict_selected_var = {'cmcah':['calendar', 'conso', 'holiday_days','meteo'], 'cmca':['calendar', 'conso','meteo']}


# In[28]:


list_nb_hidden_layers = [2,4,6,8]


# In[29]:


list_dropout = [0,0.05,0.1,0.15,0.20]


# In[32]:


training_epochs=400
batch_size=100


# In[33]:


combination = list(itertools.product(list_nb_hidden_layers, list_dropout))


# In[34]:


date_test_start = datetime.datetime(year=2016, month=6, day=11)
date_test_end = datetime.datetime(year=2017, month=6, day=10)


# In[35]:


# Prepare results wrap up 
results_df = pd.DataFrame(columns=['name', 'layer_dims','dropout_rates','batchsize',
                                           'best_iter', 'train_mse',
                                           'train_mae', 'train_mape',
                                           'test_mse', 'test_mae',
                                           'test_mape'])
path_results = path_out


# ### Main loop

# In[36]:


for gen_name, selected_variables in dict_selected_var.items():
    
    # Prepare dataset
    x_conso_selected_var = select_variables(x_conso, dict_colnames_conso, selected_variables)
    dataset, dict_ds = get_train_test_sets(x_conso_selected_var, y_conso, date_test_start, date_test_end)
    dataset = normalized_dataset(dataset, dict_colnames_conso)
    
    nb_hidden_units_base = dataset['train']['x'].shape[1]
    
    for idx, (nb_hidden_layers, dropout) in enumerate(combination):
        
        print('========================= Model {}/{} ========================='.format(idx+1, len(combination)))
        
        # Get right number of hidden units
        nb_hidden_units = int(nb_hidden_units_base*(1+dropout))
        
        # Prepare model characteristics
        name_model = 'b1_FFNN_l{}*{}_d{}*{}_{}_norm'.format(nb_hidden_units, nb_hidden_layers,
                                                            dropout, nb_hidden_layers, 
                                                            gen_name)
        
        # Compile model
        model = FeedForward(name=name_model, output=path_out, input_dim=nb_hidden_units_base, output_dim=1, 
                   l_dims=[nb_hidden_units]*nb_hidden_layers, dropout_rates=[dropout]*nb_hidden_layers,
                   loss = 'mean_squared_error', metrics = ['mape', 'mae'])
        
        # Prepare callbacks
        callbacks = []
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=150,
                                                           verbose=0, mode='auto')

        model_checkpoint = ModelCheckpoint(os.path.join(path_out, name_model, 'models', 'model-best.hdf5'),
                                                   monitor='val_loss',
                                                   verbose=0, save_best_only=True, save_weights_only=False,
                                                   mode='auto', period=1)

        tensorboard_model = TensorBoard(log_dir=os.path.join(path_out, name_model, 'results', 'logs', time.strftime('%Y-%m-%d_%H:%M', time.localtime(time.time()))))
        tensorboard_summary = TensorBoard(log_dir=os.path.join(path_out, 'logs', name_model))
        

        callbacks.append(early_stop)
        callbacks.append(model_checkpoint)
        callbacks.append(tensorboard_model)
        callbacks.append(tensorboard_summary)
        
        # Train model
        model.main_train(dataset, training_epochs=training_epochs, batch_size=batch_size, callbacks=callbacks)
        
        # Get result and put it in results
        _, result = model.analyze_history(dataset)
        
        results_df= results_df.append(result, ignore_index=True)
        results_df.to_csv(os.path.join(path_results, 'b1_results.csv'), sep=';')
    
        # Reset graph
        K.clear_session()
        import tensorflow as tf
        tf.reset_default_graph()
    

