import os
import pickle
import pandas as pd

from aed.detection import get_residuals, get_atypical_score, get_atypical_ds
from aed.atypical_event import AtypicalEventsList, get_atypical_events_list

from keras.models import load_model


def get_prediction_results(path_models_folder, dict_datasets, mode=1, type_model='keras'):
    """
    Wrap up the results given by a model fitted on different periods (K-crossvalidation)
    The information of the models must be contained in a folder that respect a formalism:
        - contain a .pickle file corresponding to the datasets infos
        - for each subperiods, contain a subfolder (with name begining by the name of the subperiod) containing the keras architecture.

    :param path_models_folder:
    :param name_dict_dataset:
    :param mode: mode of the detector
    :return: a dictionnary woth the full detector agetnd predeiction dataframe
    """

    if type(dict_datasets) == str:
        with open(os.path.join(path_models_folder, dict_datasets), 'rb') as f:
            dict_datasets = pickle.load(f)

    list_models_folder = [el for el in os.listdir(path_models_folder) if el.startswith(tuple(dict_datasets.keys()))]
    assert len(list_models_folder) == len(dict_datasets.keys())

    prediction_results = {}

    for name_period in dict_datasets.keys():
        name_model_folder = [el for el in list_models_folder if el.startswith(name_period)][0]

        # To deal with different types of models
        if type_model == 'keras':
            model = load_model(os.path.join(path_models_folder, name_model_folder, 'models', 'model-best.hdf5'))
        elif type_model == 'sklearn':
            with open(os.path.join(path_models_folder, name_model_folder,'model.pickle'),'rb') as f:
                model = pickle.load(f)

        y = dict_datasets[name_period]['dataset']['test']['y']
        y_hat = model.predict(dict_datasets[name_period]['dataset']['test']['x']).reshape(-1,1)

        prediction_df = pd.DataFrame(
            {'ds': dict_datasets[name_period]['dict_ds']['test'], 'prediction': y_hat.flatten()})
        residuals_df = get_residuals(y_obs=y, y_hat=y_hat, ds=dict_datasets[name_period]['dict_ds']['test'])
        detector_df = get_atypical_score(residuals_df, mode=mode)

        prediction_results[name_period] = {'detector_df': detector_df, 'prediction_df': prediction_df}

    del dict_datasets, model

    return prediction_results


def get_aed_results(prediction_results, threshold):
    """
    From the results of the model on all the periods and a given threshold, return the aed files.
    :param prediction_results:
    :param threshold:
    :return:
    """

    aed_results = {}

    for name_period, el in prediction_results.items():
        detector_df = el['detector_df']

        atypical_df = get_atypical_ds(detector_df, threshold)
        events_list_model = get_atypical_events_list(atypical_df, atypical_name=name_period)

        aed_results[name_period] = {'atypical_df': atypical_df, 'ael_model': events_list_model}

    # get full ae list
    ael_full_model = AtypicalEventsList()

    for name_period, el in aed_results.items():
        ael_model = el['ael_model']
        ael_full_model = ael_full_model.get_union(ael_model)

    return aed_results, ael_full_model


def get_complete_df(prediction_results, aed_results):
    """
    From the aed results, get the full dataframes containing all info
    :param prediction_results:
    :param aed_results:
    :return:
    """

    atypical_full_df = pd.DataFrame()
    prediction_full_df = pd.DataFrame()

    for name_period in prediction_results.keys():
        atypical_full_df = atypical_full_df.append(aed_results[name_period]['atypical_df'], ignore_index=True)
        prediction_full_df = prediction_full_df.append(prediction_results[name_period]['prediction_df'],
                                                       ignore_index=True)

    prediction_full_df = prediction_full_df.sort_values(by='ds', axis=0)
    atypical_full_df = atypical_full_df.sort_values(by='ds', axis=0)

    prediction_full_df = prediction_full_df.reset_index(drop=True)
    atypical_full_df = atypical_full_df.reset_index(drop=True)

    return atypical_full_df, prediction_full_df