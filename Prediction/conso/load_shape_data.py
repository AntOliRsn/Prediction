import os
import datetime
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


from aed.atypical_event import AtypicalEventsList, AtypicalEvent

def load_raw_data_conso(path_data_folder):
    """
    Shape the raw consumption data, gather them in a python dictionary and save it as a pickle file on disk.

    :param path_data_folder: Path of the folder containing all the raw files:
                -conso_Y.csv
                -meteoX_T.csv
                -joursFeries.csv
                -Tempo_history_merged.csv
    :return:
    """

    #path_data_folder = os.path.join("/local/home/antorosi/Documents/Prediction/data")

    # CONSUMPTION
    conso_csv = os.path.join(path_data_folder, "conso_Y.csv")
    conso_df = pd.read_csv(conso_csv, sep=";", engine='c', header=0)
    del conso_csv

    conso_df['ds'] = pd.to_datetime(conso_df['date'] + ' ' + conso_df['time'])

    # get only national observation
    conso_df = conso_df[['ds', 'Consommation NAT t0']]
    conso_df.columns = ['ds','conso_nat_t0']

    print('Consumption data loaded')

    # TEMPERATURE
    meteo_csv = os.path.join(path_data_folder, "meteoX_T.csv")
    meteo_df = pd.read_csv(meteo_csv, sep=";", engine='c', header=0)
    meteo_df['ds'] = pd.to_datetime(meteo_df['date'] + ' ' + meteo_df['time'])
    del meteo_csv

    # USELESS NOW
    # time_decay = (conso_df.ds.iloc [-1] - meteo_df.ds.iloc[-1]).seconds/(60*15)
    #
    # if time_decay > 10:
    #     print("meteo time series length does't math the conso one")
    #     return
    # else:
    #     ref = meteo_df.iloc[-1]
    #     for i in range(int(time_decay)):
    #         meteo_df.append(ref)

    # Drop the duplicates (likely due to the change of hour)

    # Correct the last two timestamp manually (like manually manually)
    # ts_ref = meteo_df.ds.iloc[-3]
    # ts_2 = ts_ref + datetime.timedelta(minutes=15)
    # ts_1 = ts_2 + datetime.timedelta(minutes=15)
    # meteo_df.loc[meteo_df.index.values[-2],'ds'] = ts_2
    # meteo_df.loc[meteo_df.index.values[-1],'ds'] = ts_1

    meteo_df = meteo_df.drop_duplicates(subset='ds',keep='last')

    # get observation only
    meteo_df = meteo_df[list(meteo_df.columns[meteo_df.columns.str.endswith('Th+0')]) + ['ds']]
    stationColumns = meteo_df.columns[list(meteo_df.columns.str.endswith('Th+0'))]
    meteo_df['meteo_natTh+0'] = meteo_df[stationColumns].mean(axis=1)

    print('Meteo data loaded')

    # HOLIDAY DAYS
    holiday_days_csv = os.path.join(path_data_folder, "joursFeries.csv")
    holiday_days_df = pd.read_csv(holiday_days_csv, sep=";")
    holiday_days_df.ds = pd.to_datetime(holiday_days_df.ds)

    # Putting dayly label to hourly label
    # TODO: Find a better, vectorized solution
    day = holiday_days_df.ds[0]
    start_day = day
    end_day = day + pd.DateOffset(hour=23) + pd.DateOffset(minute=45)
    day_hourly = pd.date_range(start_day, end_day, freq='15min')

    for day in holiday_days_df.ds[1:]:
        start_day = day
        end_day = day + pd.DateOffset(hour=23) + pd.DateOffset(minute=45)
        day_hourly = day_hourly.append(pd.date_range(start_day, end_day, freq='15min'))

    day_hourly.name = 'ds'
    holiday_days_df = holiday_days_df.set_index('ds')
    holiday_days_df = holiday_days_df.reindex(day_hourly, method="ffill")
    holiday_days_df = holiday_days_df.reset_index()

    holiday_days_df.columns = ['ds', 'type_holiday']

    print('Holiday Days data loaded')

    # TEMPO DAYS
    tempo_csv = os.path.join(path_data_folder, "Tempo_history_merged.csv")
    tempo_df = pd.read_csv(tempo_csv, sep=";")
    tempo_df['ds'] = pd.to_datetime(tempo_df.Date)
    tempo_df.drop(['Date'], axis=1, inplace=True)

    # Putting dayly label to hourly label
    start_day = min(tempo_df.ds)
    end_day = max(tempo_df.ds) + pd.DateOffset(hour=23) + pd.DateOffset(minute=45)
    hourly_tf = pd.date_range(start_day,end_day,freq='15min')

    hourly_tf.name='ds'
    tempo_df = tempo_df.set_index('ds')
    tempo_df = tempo_df.reindex(hourly_tf, method='ffill')
    tempo_df = tempo_df.reset_index()

    tempo_df.columns =  ['ds', 'type_tempo']

    print('Tempo data loaded')

    # Gathering data into dictionnary
    dict_data_conso = {'conso':conso_df, 'meteo':meteo_df, 'holiday_days':holiday_days_df, 'tempo':tempo_df}

    # Saving dict
    with open(os.path.join(path_data_folder, 'dict_data_conso.pickle'), 'wb') as file:
        pickle.dump(dict_data_conso, file, protocol=pickle.HIGHEST_PROTOCOL)

    print('Shaped data saved in {}'.format(os.path.join(path_data_folder, 'dict_data_conso.pickle')))


def load_data_conso(path_data_folder):
    """
    Load a dictionnary containing all the needed data related to consumption prediction

    :param path_data_folder: path of the folder containing the data
    :return: python dictoinnary
    """

    # Checking if the dict containing conso data already exists

    if not os.path.exists(os.path.join(path_data_folder, 'dict_data_conso.pickle')):
        load_raw_data_conso(path_data_folder)

    with open(os.path.join(path_data_folder, 'dict_data_conso.pickle'), 'rb') as f:
        dict_data_conso = pickle.load(f)

    return dict_data_conso


def get_uniformed_data_conso(dict_data_conso):
    """
    Put the data from dict_data_conso in the same dataframe.
    Allows to 'uniform' the data as depending on the source some days or hours are skipped (mostly du to the change of hours).
    The taken reference is the time series from the consumption.

    :param dict_data_conso:
    :return:
    """

    dict_colnames_conso = {}

    for key, df in dict_data_conso.items():
        dict_colnames_conso[key] = [el for el in df.columns if el!='ds']

    data_conso_df = dict_data_conso['conso']. copy()
    data_conso_df = pd.merge(data_conso_df, dict_data_conso['meteo'], on='ds', how='left')

    # Deal with the change of hour
    ds = data_conso_df.ds
    ds_shifted = ds.shift(1)
    ds_shifted[0] = ds[0] - datetime.timedelta(minutes=15)
    mask = ds - ds_shifted == datetime.timedelta(hours=1, minutes=15)

    for row in data_conso_df[mask].iterrows():
        index = row[0]
        serie = row[1]
        serie_before = data_conso_df.iloc[index-1]

        ds_interpolate = pd.date_range(serie_before['ds'], serie['ds'], freq='15min')
        ds_interpolate = ds_interpolate[1:-1]

        interploate_df = pd.DataFrame(np.nan,index=range(len(ds_interpolate)),columns=data_conso_df.columns)
        interploate_df['ds'] = ds_interpolate

        data_conso_df = data_conso_df.append(interploate_df)

    data_conso_df = data_conso_df.sort_values('ds')
    data_conso_df = data_conso_df.set_index('ds')
    data_conso_df = data_conso_df.interpolate('linear')

    data_conso_df = data_conso_df.reset_index()

    # check that change of hour were the only reason for missing values
    ds = data_conso_df.ds
    ds_shifted = ds.shift(1)
    ds_shifted[0] = ds[0] - datetime.timedelta(minutes=15)
    assert len(set(ds - ds_shifted)) == 1

    ####

    data_conso_df = pd.merge(data_conso_df, dict_data_conso['tempo'], on='ds', how='left')

    # formating holiday days to be boolean
    hd_ds = dict_data_conso['holiday_days'].copy()
    hd_ds['is_holiday_day'] = np.array(hd_ds['type_holiday']).astype('bool').astype('int')
    data_conso_df = pd.merge(data_conso_df, hd_ds[['ds','is_holiday_day']], on='ds', how='left')
    pd.set_option('chained_assignment', None) # To avoid message about chain assignment, necessary here
    data_conso_df['is_holiday_day'].loc[data_conso_df['is_holiday_day'].isna()] = 0
    pd.set_option('chained_assignment', 'warn')

    dict_colnames_conso['holiday_days'] = ['is_holiday_day']

    if 'atypical_events' in dict_data_conso.keys():
        ae_ds = dict_data_conso['atypical_events'].copy()
        data_conso_df = pd.merge(data_conso_df, ae_ds, on='ds', how='left')
        pd.set_option('chained_assignment', None)  # To avoid message about chained assignment, necessary here
        data_conso_df['is_atypical'].loc[data_conso_df['is_atypical'].isna()] = 0
        pd.set_option('chained_assignment', 'warn')

        dict_colnames_conso['atypical_events'] = ['is_atypical']

    return data_conso_df, dict_colnames_conso


def change_granularity(data_conso_df, granularity = "1H"):

    if granularity not in ["1H", "15min", "30min"]:
        print('"granularity" must be in ["1H", "15min", "30min"]')
        return

    minutes = np.array(data_conso_df.ds.dt.minute)

    if granularity == "1H":
        mask = np.where(minutes == 0)[0]
    if granularity == "30min":
        mask = np.where((minutes == 30) | (minutes == 0))[0]
    if granularity == "15min":
        mask = np.array(data_conso_df.index)

    data_conso_new_granu_df = data_conso_df.loc[mask].copy()
    data_conso_new_granu_df = data_conso_new_granu_df.reset_index(drop=True)

    return data_conso_new_granu_df


def get_x_y_prediction_conso(data_conso_df, dict_colnames_conso, lags=[24]):

    # Get one hot encoding of calendar informations (hour, day, month)
    timeserie = data_conso_df.ds
    weekday = timeserie.dt.weekday
    month = timeserie.dt.month
    hour = timeserie.dt.hour
    minute = timeserie.dt.minute

    calendar_ds = pd.DataFrame({'month': month, 'weekday': weekday, 'hour': hour, 'minute':minute ,'ds': timeserie})

    # One hot encoding
    encoded_weekday = pd.get_dummies(calendar_ds['weekday'], prefix="weekday")
    encoded_month = pd.get_dummies(calendar_ds['month'], prefix="month")
    encoded_hour = pd.get_dummies(calendar_ds['hour'], prefix="hour")
    encoded_minute = pd.get_dummies(calendar_ds['minute'], prefix="minute")

    # Check time_step
    timedelta = (timeserie[1] - timeserie[0]).seconds/(60*15)
    nb_columns_encoded_minute = encoded_minute.shape[1]

    expected_dim = {4:1, 2:2, 1:4}
    assert expected_dim[nb_columns_encoded_minute] == timedelta

    if  nb_columns_encoded_minute == 1:
        calendar_encoded_ds = pd.concat([encoded_weekday, encoded_month, encoded_hour, timeserie], axis=1)
    else:
        calendar_encoded_ds = pd.concat([encoded_weekday, encoded_month, encoded_hour, encoded_minute, timeserie], axis=1)

    dict_colnames_conso['calendar'] = [el for el in calendar_encoded_ds.columns if el !='ds']

    # Merge conso and meteo
    x_conso = pd.merge(data_conso_df, calendar_encoded_ds, on='ds', how='left')
    x_conso = x_conso.drop('type_tempo', axis = 1)

    # Add lag

    # No need to put calendar variables in lags
    mask_columns_lag = ['ds'] + dict_colnames_conso['conso'] + dict_colnames_conso['meteo'] + dict_colnames_conso['holiday_days']
    if 'atypical_events' in dict_colnames_conso.keys():
        mask_columns_lag += dict_colnames_conso['atypical_events']

    for lag in list(lags):

        shift_length = nb_columns_encoded_minute * lag

        # Positive lag
        x_conso_plus = x_conso[mask_columns_lag].copy()
        x_conso_plus.columns = ["{}_plus_{}H".format(el, lag) for el in x_conso_plus.columns]
        x_conso_plus = x_conso_plus.rename(columns={"ds_plus_{}H".format(lag) : 'ds'})
        mask_col = [el for el in x_conso_plus.columns if el !='ds']
        x_conso_plus[mask_col] = x_conso_plus[mask_col].shift(-shift_length)

        # Negative lag
        x_conso_minus = x_conso[mask_columns_lag].copy()
        x_conso_minus.columns = ["{}_minus_{}H".format(el, lag) for el in x_conso_minus.columns]
        x_conso_minus = x_conso_minus.rename(columns={"ds_minus_{}H".format(lag): 'ds'})
        mask_col = [el for el in x_conso_minus.columns if el != 'ds']
        x_conso_minus[mask_col] = x_conso_minus[mask_col].shift(shift_length)

        # Merge
        x_conso = pd.merge(x_conso, x_conso_minus, on='ds', how='left')
        x_conso = pd.merge(x_conso, x_conso_plus, on='ds', how='left')

    # Drop points with Na due to the positive and negative lags
    x_conso = x_conso[x_conso.isna().sum(axis=1) == 0]
    x_conso = x_conso.reset_index(drop=True)

    # Get x and y
    y_conso = x_conso[['ds', 'conso_nat_t0']]
    x_conso = x_conso.drop('conso_nat_t0', axis=1)

    return x_conso, y_conso, dict_colnames_conso


def select_variables(x_conso, dict_colnames_conso, list_variable):
    assert set(list_variable).issubset(set(dict_colnames_conso.keys()))

    mask = ['ds']
    for variable in list_variable:
        mask_variable = [el for el in x_conso.columns if el.startswith(tuple(dict_colnames_conso[variable]))]
        mask += mask_variable

    sorted_mask = [el for el in x_conso if el in mask]

    x_conso_selected_variables = x_conso[sorted_mask].copy()

    return x_conso_selected_variables


def get_train_test_sets(x_conso, y_conso, date_test_start, date_test_end):
    """
    split the data set in train and test set

    :param x_conso: dataframe
    :param y_conso: dataframe
    :param date_test_start: timestamp of the first day of the test set
    :param date_test_end: timestamp of the last day of the test set
    :return: dataset: dictionary containing the train and test set (x and y)
             dict_ds: dictionary containing the time series of the train and test set
    """

    mask_test = (x_conso.ds >= date_test_start) & (x_conso.ds < date_test_end + datetime.timedelta(days=1))

    x_test = x_conso[mask_test]
    y_test = y_conso[mask_test]
    x_train = x_conso[np.invert(mask_test)]
    y_train = y_conso[np.invert(mask_test)]

    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    dict_ds = {'train': x_train.ds, 'test': x_test.ds}

    x_test = x_test.drop('ds', axis=1)
    y_test = y_test.drop('ds', axis=1)
    x_train = x_train.drop('ds', axis=1)
    y_train = y_train.drop('ds', axis=1)

    dataset = {}
    dataset['train'] = {'x': x_train, 'y': y_train}
    dataset['test'] = {'x': x_test, 'y': y_test}

    return dataset, dict_ds


def normalized_dataset(dataset, dict_colnames_conso):
    """
    Normalization of the needed columns

    :param dataset:
    :param dict_colnames_conso:
    :return: dataset_scaled
    """
    x_train = dataset['train']['x']
    x_test = dataset['test']['x']

    # Getting columns to normalized
    mask_conso = [el for el in x_train.columns if el.startswith(tuple(dict_colnames_conso['conso']))]
    mask_meteo = [el for el in x_train.columns if el.startswith(tuple(dict_colnames_conso['meteo']))]

    cols_to_normalized = mask_conso + mask_meteo

    # Fitting scaler on train
    scaler = StandardScaler(with_mean=True, with_std=True)
    scalerfit = scaler.fit(x_train[cols_to_normalized])

    # Applying filter on train
    cols_normalized = scalerfit.transform(x_train[cols_to_normalized])

    x_train_scaled = x_train.copy()
    for i, col_name in enumerate(cols_to_normalized):
        x_train_scaled[col_name] = cols_normalized[:, i]

    # Applying filter on test
    cols_normalized = scalerfit.transform(x_test[cols_to_normalized])

    x_test_scaled = x_test.copy()
    for i, col_name in enumerate(cols_to_normalized):
        x_test_scaled[col_name] = cols_normalized[:, i]

    dataset_scaled = dataset
    dataset_scaled['train']['x'] = x_train_scaled
    dataset_scaled['test']['x'] = x_test_scaled

    return dataset_scaled


def add_atypical_events_to_dict_data_conso(dict_data_conso, ael):

    ae_df = ael.get_events_list()[['date_start', 'is_atypical']]
    ae_df.rename(columns={"date_start":"ds"}, inplace=True)

    # Way to init a DateTimeIndex ... I didn't find an other method
    ae_period = pd.DatetimeIndex(start=datetime.datetime(2142,12,12), end=datetime.datetime(2142,12,12),freq="15min")

    for ae in ael.set_atypical_events:
        start_day = ae.date_start
        end_day = ae.date_end
        ae_period = ae_period.append(pd.date_range(start_day, end_day, freq='15min'))

    ae_period = ae_period.drop(labels=datetime.datetime(2142,12,12))
    ae_period = ae_period.sort_values()

    ae_period.name = 'ds'
    ae_df = ae_df.set_index('ds')
    ae_df = ae_df.reindex(ae_period, method="ffill")
    ae_df = ae_df.reset_index()

    dict_data_conso['atypical_events'] = ae_df

    return dict_data_conso


