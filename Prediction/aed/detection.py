import os
import numpy as np
import pandas as pd
from scipy import stats
import datetime

import plotly.offline as py
import plotly.graph_objs as go
from plotly import tools

from aed.atypical_event import AtypicalEvent, AtypicalEventsList

def get_residuals(y_obs, y_hat, ds):
    """

    :param y_obs:
    :param y_hat:
    :param ds:
    :return:
    """

    residuals = np.array(y_obs - y_hat)
    residuals = {'ds':ds, 'residuals':residuals.flatten()}
    residuals_ds = pd.DataFrame(residuals)

    return residuals_ds


def get_atypical_score(residuals_df, mode = 1, merge=True):
    """

    :param residuals_df:
    :param merge:
    :return:
    """

    residuals_df_detector = residuals_df.copy()

    diff_days = (residuals_df_detector.ds - residuals_df_detector.ds[0]).apply(lambda td: td.days)
    residuals_df_detector['diff_days'] = diff_days

    # Daily mean of residuals
    if mode == 1:
        groupe_day = residuals_df_detector.groupby(['diff_days'])

        dayly_indic = groupe_day['residuals'].aggregate([np.mean])
        dayly_indic.reset_index(inplace=True)


        dayly_indic['z_score'] = stats.zscore(dayly_indic['mean'])
        dayly_indic['a_score'] = 1-2 * stats.norm.cdf(-np.abs(dayly_indic['z_score']))

        detector_ds = pd.merge(residuals_df_detector, dayly_indic, how='left', on='diff_days')
        #detector_ds = detector_ds.drop(['mean', 'z_score', 'diff_days'], axis=1)
        detector_ds = detector_ds.drop(['mean', 'diff_days'], axis=1)

    # Daily mean of residuals absolute values
    if mode == 2:
        residuals_df_abs_detector = residuals_df_detector.copy()
        residuals_df_abs_detector['residuals'] = residuals_df_detector['residuals'].apply(lambda el: np.abs(el))

        groupe_day = residuals_df_abs_detector.groupby(['diff_days'])

        dayly_indic = groupe_day['residuals'].aggregate([np.mean])
        dayly_indic.reset_index(inplace=True)

        dayly_indic['z_score'] = stats.zscore(dayly_indic['mean'])
        dayly_indic['a_score'] = 1 - 2 * stats.norm.cdf(-np.abs(dayly_indic['z_score']))

        detector_ds = pd.merge(residuals_df_detector, dayly_indic, how='left', on='diff_days')
        #detector_ds = detector_ds.drop(['mean', 'z_score', 'diff_days'], axis=1)
        detector_ds = detector_ds.drop(['mean', 'diff_days'], axis=1)

    # Multivariate normal distribution fitted
    if mode == 3:
        # Getting (-1,24) dim matrix
        residuals_dist_df = residuals_df_detector.copy()
        residuals_dist_df['minutes'] = residuals_dist_df['ds'].dt.hour * 100 + residuals_dist_df['ds'].dt.minute

        residuals_array = residuals_dist_df[['diff_days', 'minutes', 'residuals']].pivot('diff_days', 'minutes')
        residuals_array[residuals_array.isna()] = residuals_array.as_matrix().mean(axis=0)[7]
        residuals_array = residuals_array.as_matrix()

        # Fitting 24 multinomial normal distribution
        mean = residuals_array.mean(axis=0)
        cov = np.cov(residuals_array, rowvar=False)
        dist = stats.multivariate_normal(mean=mean, cov=cov)

        # getting the pvalue of each day
        pvalue = dist.pdf(residuals_array)

        # log regularization
        score = -np.log(pvalue / pvalue.max())

        # normalization
        min_p = score.min()
        max_p = score.max()
        score = (score - min_p) / (max_p - min_p)

        # to dataframe
        score_df = pd.DataFrame({'a_score': score, 'z_score': pvalue, 'diff_days': diff_days.unique()})

        detector_ds = pd.merge(residuals_df_detector, score_df, how='left', on='diff_days')
        detector_ds = detector_ds.drop(['diff_days'], axis=1)

    if mode == 4:
        groupe_day = residuals_df_detector.groupby(['diff_days'])

        dayly_indic = groupe_day['residuals'].aggregate([np.mean])
        dayly_indic.reset_index(inplace=True)


        dayly_indic['z_score'] = stats.zscore(dayly_indic['mean'])
        dayly_indic['a_score'] = 1-2 * stats.norm.pdf(dayly_indic['z_score'])

        detector_ds = pd.merge(residuals_df_detector, dayly_indic, how='left', on='diff_days')
        #detector_ds = detector_ds.drop(['mean', 'z_score', 'diff_days'], axis=1)
        detector_ds = detector_ds.drop(['mean', 'diff_days'], axis=1)

    if mode == 5:
        residuals_df_abs_detector = residuals_df_detector.copy()
        residuals_df_abs_detector['residuals'] = residuals_df_detector['residuals'].apply(lambda el: np.abs(el))

        groupe_day = residuals_df_abs_detector.groupby(['diff_days'])

        dayly_indic = groupe_day['residuals'].aggregate([np.mean])
        dayly_indic.reset_index(inplace=True)

        dayly_indic['z_score'] = stats.zscore(dayly_indic['mean'])
        dayly_indic['a_score'] = 1 - 2 * stats.norm.pdf(dayly_indic['z_score'])

        detector_ds = pd.merge(residuals_df_detector, dayly_indic, how='left', on='diff_days')
        # detector_ds = detector_ds.drop(['mean', 'z_score', 'diff_days'], axis=1)
        detector_ds = detector_ds.drop(['mean', 'diff_days'], axis=1)

    if not merge:
        detector_ds = detector_ds.drop(['residuals'], axis=1)

    return detector_ds


def get_atypical_ds(detector_ds, threshold, merge=True):
    """

    :param detector_ds:
    :param threshold:
    :param merge:
    :return:
    """

    atypical_ds = detector_ds.copy()

    mask_event = atypical_ds['a_score'] > threshold
    atypical_ds['is_atypical'] = np.array(mask_event).astype('int')

    if not merge:
        atypical_ds.drop(['a_score'], axis=1)

    return atypical_ds


def get_next_hd_events(atypical_new_df, ael_hd_reference):

    ael_reference_df = ael_hd_reference.get_events_list().copy()
    ael_reference_df = ael_reference_df.rename(index=str, columns={"date_start": "ds"})

    mask = ael_reference_df['type_event'] == 1
    ael_reference_df = ael_reference_df[mask]

    results_df = pd.merge(atypical_new_df, ael_reference_df[['ds', 'type_event']], on='ds', how='left')

    results_df = results_df.set_index('ds', drop=True)

    mask = results_df['type_event'] == 1

    index_plus_one = results_df[mask].index + datetime.timedelta(days=1)
    index_minus_one = results_df[mask].index - datetime.timedelta(days=1)

    for el in list(set(index_minus_one) - set(results_df.index)):
        idx = index_minus_one.get_loc(el)
        index_minus_one = index_minus_one.delete(idx)

    for el in list(set(index_plus_one) - set(results_df.index)):
        idx = index_plus_one.get_loc(el)
        index_plus_one = index_plus_one.delete(idx)

    mask_minus = results_df.loc[index_minus_one, 'type_event'].isna()
    index_minus_one_true = results_df.loc[index_minus_one][mask_minus].index
    results_df.loc[index_minus_one_true, 'type_event'] = -1

    mask_plus = results_df.loc[index_plus_one, 'type_event'].isna()
    index_plus_one_true = results_df.loc[index_plus_one][mask_plus].index
    results_df.loc[index_plus_one_true, 'type_event'] = -1

    results_df = results_df.reset_index()

    return results_df


def sort_atypical_events(atypical_df, ael_hd_reference=None):
    """
    return the dayly event sorted according to their atypical scores

    :param atypical_df:
    :return:
    """

    mask = (atypical_df.ds.dt.hour == 0) & (atypical_df.ds.dt.minute == 0)

    atypical_new_df = atypical_df[mask].copy()

    atypical_new_df = atypical_new_df.sort_values(by='a_score', axis=0, ascending=False)
    atypical_new_df = atypical_new_df.reset_index(drop=True)

    if ael_hd_reference is not None:
        atypical_new_df = get_next_hd_events(atypical_new_df, ael_hd_reference)

    return atypical_new_df


def prediction_conso_plot(data_conso_df, atypical_ds, prediction_ds, dict_colnames_conso, path_plot, name):
    """

    :param data_conso_df:
    :param atypical_ds:
    :param prediction_ds:
    :param dict_colnames_conso:
    :param path_plot:
    :param name:
    :return:
    """

    # The code is messy but working ...
    # Due to difficulties with plotly library
    # Some element can't be used because they are not 'JSON serializable'

    # merge info
    data_conso_df_period = pd.merge(data_conso_df, atypical_ds, on='ds', how='right')

    # Calendar plot
    tempo_df = data_conso_df_period[['ds', 'is_holiday_day', 'type_tempo']].copy()
    hours = tempo_df.ds.dt.hour
    mask_hour = np.where(hours == 12)
    tempo_df = tempo_df.loc[mask_hour]
    tempo_df = tempo_df.reset_index(drop=True)

    colour_file_none = 'rgba(214, 138, 0, '
    colour_line_none = 'rgba(214, 138, 0, '
    colour_file_blue = 'rgba(55, 128, 191, '
    colour_line_blue = 'rgba(55, 128, 191, '
    colour_file_red = 'rgba(219, 64, 82, '
    colour_line_red = 'rgba(219, 64, 82, '
    colour_file_white = 'rgba(50, 171, 96, '
    colour_line_white = 'rgba(50, 171, 96, '

    tempo_df['colour_file'] = colour_file_none
    tempo_df['colour_line'] = colour_line_none

    mask = np.where(tempo_df['type_tempo'] == 'BLEU')[0]
    tempo_df.loc[mask, 'colour_file'] = colour_file_blue
    tempo_df.loc[mask, 'colour_line'] = colour_line_blue

    mask = np.where(tempo_df['type_tempo'] == 'ROUGE')[0]
    tempo_df.loc[mask, 'colour_file'] = colour_file_red
    tempo_df.loc[mask, 'colour_line'] = colour_line_red

    mask = np.where(tempo_df['type_tempo'] == 'BLANC')[0]
    tempo_df.loc[mask, 'colour_file'] = colour_file_white
    tempo_df.loc[mask, 'colour_line'] = colour_line_white

    mask_weekend = (tempo_df.ds.dt.weekday == 5) | (tempo_df.ds.dt.weekday == 6)
    tempo_df.loc[mask_weekend, 'colour_line'] = 'rgba(0,0,0, '
    tempo_df.loc[mask_weekend, 'type_tempo'] = 1

    tempo_df['type_tempo'] = 1

    mask = np.where(tempo_df['is_holiday_day'] == 1)[0]
    tempo_df.loc[mask, 'colour_file'] += '1.0)'
    mask = np.where(tempo_df['is_holiday_day'] != 1)[0]
    tempo_df.loc[mask, 'colour_file'] += '0.4)'
    tempo_df.colour_line += '0.7)'

    trace_calendar = go.Bar(
        x=tempo_df.ds,
        y=tempo_df['type_tempo'],
        width=60000000 * np.ones(len(tempo_df.ds)),
        marker=dict(
            color=tempo_df.colour_file,
            line=dict(
                color=tempo_df.colour_line,
                width=2
            )
        )
    )

    # Meteo plot
    trace_observation_meteo = go.Scatter(
        x=data_conso_df_period.ds,
        y=data_conso_df_period['meteo_natTh+0'],
        name='Temperature Observation',
        fill='tozeroy',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=1
        )
    )

    # Conso Plot
    trace_observation_conso = go.Scatter(
        x=data_conso_df_period.ds,
        y=data_conso_df_period['conso_nat_t0'],
        fill='tozeroy',
        name="Consumption observation",
        line=dict(
            color=('rgb(46, 178, 186)'),
            width=1
        )
    )

    trace_prediction_conso = go.Scatter(
        x=prediction_ds.ds,
        y=prediction_ds['prediction'],
        name='Consumption Prediction',
        line=dict(
            color=('rgb(6, 178, 186)'),
            width=2,
            dash='dash')
    )

    trace_error_conso = go.Scatter(
        x=atypical_ds.ds,
        y=atypical_ds['residuals'],
        name='Consumption Forecasting error',
        line=dict(
            color=('rgb(255, 188, 56)'),
            width=1
        )
    )

    mask = np.where(atypical_ds['is_atypical'] == 1)[0]
    trace_atypical_points_TP = go.Scatter(
        x=atypical_ds.loc[mask, 'ds'],
        y=atypical_ds.loc[mask, 'residuals'],
        name='atypical_TP',
        mode='markers'
    )

    mask = np.where(atypical_ds['is_atypical'] == 0)[0]
    trace_atypical_points_FP = go.Scatter(
        x=atypical_ds.loc[mask, 'ds'],
        y=atypical_ds.loc[mask, 'residuals'],
        name='atypical_FP',
        mode='markers'
    )

    mask = np.where(atypical_ds['is_atypical'] == -1)[0]
    trace_atypical_points_FN = go.Scatter(
        x=atypical_ds.loc[mask, 'ds'],
        y=atypical_ds.loc[mask, 'residuals'],
        name='atypical_FN',
        mode='markers'
    )

    # Final plot
    fig = tools.make_subplots(rows=4, cols=1,
                              shared_xaxes=True, shared_yaxes=False,
                              vertical_spacing=0.1,
                              )

    fig.append_trace(trace_atypical_points_TP, 2, 1)
    fig.append_trace(trace_atypical_points_FP, 2, 1)
    fig.append_trace(trace_atypical_points_FN, 2, 1)
    fig.append_trace(trace_observation_conso, 1, 1)
    fig.append_trace(trace_prediction_conso, 1, 1)
    fig.append_trace(trace_error_conso, 2, 1)
    fig.append_trace(trace_observation_meteo, 3, 1)
    fig.append_trace(trace_calendar, 4, 1)

    fig['layout'].update(title='Forecast analysis')

    fig['layout']['yaxis1'].update(title='Consumption [MWh]')
    fig['layout']['yaxis2'].update(title='Error [MWh]')
    fig['layout']['yaxis3'].update(title='Temperature [°C]')
    fig['layout']['yaxis4'].update(title='Calendar Event')
    fig['layout']['xaxis4'].update(title='Time')

    return (py.plot(fig, filename=os.path.join(path_plot, name)))


def aed_conso_plot(data_conso_df, atypical_ds, prediction_ds, dict_colnames_conso, path_plot, name):
    """

    :param data_conso_df:
    :param atypical_ds:
    :param prediction_ds:
    :param dict_colnames_conso:
    :param path_plot:
    :param name:
    :return:
    """

    # The code is messy but working ...
    # Due to difficulties with plotly library
    # Some element can't be used because they are not 'JSON serializable'

    # merge info
    data_conso_df_period = pd.merge(data_conso_df, atypical_ds, on='ds', how='right')

    # Calendar plot
    tempo_df = data_conso_df_period[['ds', 'is_holiday_day', 'type_tempo']].copy()
    hours = tempo_df.ds.dt.hour
    mask_hour = np.where(hours == 12)
    tempo_df = tempo_df.loc[mask_hour]
    tempo_df = tempo_df.reset_index(drop=True)

    colour_file_none = 'rgba(214, 138, 0, '
    colour_line_none = 'rgba(214, 138, 0, '
    colour_file_blue = 'rgba(55, 128, 191, '
    colour_line_blue = 'rgba(55, 128, 191, '
    colour_file_red = 'rgba(219, 64, 82, '
    colour_line_red = 'rgba(219, 64, 82, '
    colour_file_white = 'rgba(50, 171, 96, '
    colour_line_white = 'rgba(50, 171, 96, '

    tempo_df['colour_file'] = colour_file_none
    tempo_df['colour_line'] = colour_line_none

    mask = np.where(tempo_df['type_tempo'] == 'BLEU')[0]
    tempo_df.loc[mask, 'colour_file'] = colour_file_blue
    tempo_df.loc[mask, 'colour_line'] = colour_line_blue

    mask = np.where(tempo_df['type_tempo'] == 'ROUGE')[0]
    tempo_df.loc[mask, 'colour_file'] = colour_file_red
    tempo_df.loc[mask, 'colour_line'] = colour_line_red

    mask = np.where(tempo_df['type_tempo'] == 'BLANC')[0]
    tempo_df.loc[mask, 'colour_file'] = colour_file_white
    tempo_df.loc[mask, 'colour_line'] = colour_line_white

    mask_weekend = (tempo_df.ds.dt.weekday == 5) | (tempo_df.ds.dt.weekday == 6)
    tempo_df.loc[mask_weekend, 'colour_line'] = 'rgba(0,0,0, '
    tempo_df.loc[mask_weekend, 'type_tempo'] = 1

    tempo_df['type_tempo'] = 1

    mask = np.where(tempo_df['is_holiday_day'] == 1)[0]
    tempo_df.loc[mask, 'colour_file'] += '1.0)'
    mask = np.where(tempo_df['is_holiday_day'] != 1)[0]
    tempo_df.loc[mask, 'colour_file'] += '0.4)'
    tempo_df.colour_line += '0.7)'

    trace_calendar = go.Bar(
        x=tempo_df.ds,
        y=tempo_df['type_tempo'],
        width=60000000 * np.ones(len(tempo_df.ds)),
        marker=dict(
            color=tempo_df.colour_file,
            line=dict(
                color=tempo_df.colour_line,
                width=2
            )
        )
    )

    # atypical score plot
    trace_observation_meteo = go.Scatter(
        x=data_conso_df_period.ds,
        y=data_conso_df_period['a_score'],
        name='Atypical score',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=1
        )
    )

    # Conso Plot
    trace_observation_conso = go.Scatter(
        x=data_conso_df_period.ds,
        y=data_conso_df_period['conso_nat_t0'],
        fill='tozeroy',
        name="Consumption observation",
        line=dict(
            color=('rgb(46, 178, 186)'),
            width=1
        )
    )

    trace_prediction_conso = go.Scatter(
        x=prediction_ds.ds,
        y=prediction_ds['prediction'],
        name='Consumption Prediction',
        line=dict(
            color=('rgb(6, 178, 186)'),
            width=2,
            dash='dash')
    )

    trace_error_conso = go.Scatter(
        x=atypical_ds.ds,
        y=atypical_ds['residuals'],
        name='Consumption Forecasting error',
        line=dict(
            color=('rgb(255, 188, 56)'),
            width=1
        )
    )

    mask = np.where(atypical_ds['is_atypical'] == 1)[0]
    trace_atypical_points_TP = go.Scatter(
        x=atypical_ds.loc[mask, 'ds'],
        y=atypical_ds.loc[mask, 'residuals'],
        name='atypical_TP',
        mode='markers'
    )

    mask = np.where(atypical_ds['is_atypical'] == 0)[0]
    trace_atypical_points_FP = go.Scatter(
        x=atypical_ds.loc[mask, 'ds'],
        y=atypical_ds.loc[mask, 'residuals'],
        name='atypical_FP',
        mode='markers'
    )

    mask = np.where(atypical_ds['is_atypical'] == -1)[0]
    trace_atypical_points_FN = go.Scatter(
        x=atypical_ds.loc[mask, 'ds'],
        y=atypical_ds.loc[mask, 'residuals'],
        name='atypical_FN',
        mode='markers'
    )

    # Final plot
    fig = tools.make_subplots(rows=4, cols=1,
                              shared_xaxes=True, shared_yaxes=False,
                              vertical_spacing=0.1,
                              )

    fig.append_trace(trace_atypical_points_TP, 2, 1)
    fig.append_trace(trace_atypical_points_FP, 2, 1)
    fig.append_trace(trace_atypical_points_FN, 2, 1)
    fig.append_trace(trace_observation_conso, 1, 1)
    fig.append_trace(trace_prediction_conso, 1, 1)
    fig.append_trace(trace_error_conso, 2, 1)
    fig.append_trace(trace_observation_meteo, 3, 1)
    fig.append_trace(trace_calendar, 4, 1)

    fig['layout'].update(title='Forecast analysis')

    fig['layout']['yaxis1'].update(title='Consumption [MWh]')
    fig['layout']['yaxis2'].update(title='Error [MWh]')
    fig['layout']['yaxis3'].update(title='Temperature [°C]')
    fig['layout']['yaxis4'].update(title='Calendar Event')
    fig['layout']['xaxis4'].update(title='Time')

    return (py.plot(fig, filename=os.path.join(path_plot, name)))