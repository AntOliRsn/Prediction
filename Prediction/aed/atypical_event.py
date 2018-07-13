import datetime
import pandas as pd
import numpy as np
import copy



class AtypicalEvent:
    """
    An atypical event is characterized by its name, its starting and ending date and its type.
    In order ton be consistent the rules are as follow:
    - date_start is the first timestep INCLUDED in the event
    - date_end is the last timestep INCLUDED in the event

    """

    def __init__(self, name, date_start, date_end, type_event):
        """
        :param name: int
        :param date_start: datetime or Timestamp object
        :param date_end: datetime or Timestamp object
        :param type_event: int
        """

        if type(date_start) != datetime.datetime:
            date_start = date_start.to_pydatetime()
        if type(date_end) != datetime.datetime:
            date_end = date_end.to_pydatetime()

        self.name = str(name)
        self.date_start = date_start
        self.date_end = date_end
        self.duration = date_end - date_start
        self.is_atypical = 1
        self.type_event = int(type_event)

    def get_info(self):
        """
        Return the event characteristics in a python dictionary

        :return: dictionary of characteristics
        """
        # REMARK: it would be possible to use AtypicalEvent.__dict__,
        # but we'll stick to this solution if more info need to be added later

        dict_event_info = {}
        dict_event_info['name'] = self.name
        dict_event_info['date_start'] = self.date_start
        dict_event_info['date_end'] = self.date_end
        dict_event_info['duration'] = self.duration
        dict_event_info['type_event'] = self.type_event
        dict_event_info['is_atypical'] = self.is_atypical

        return dict_event_info

    def set_isatypical(self, value):

        value = int(value)
        assert (value == 0) or (value == 1) or (value == -1)

        self.is_atypical = value

    def __eq__(self, other):
        """
        Check whereas two events are STRICTLY the same i.e same date_start, date_end and type

        :param other:
        :return:
        """

        same_ae = True

        if (self.date_start != other.date_start) or \
           (self.date_end != other.date_end) or \
           (self.type_event != other.type_event):

            same_ae = False

        return same_ae

    def __hash__(self):
        """
        hash method to allow the object to be 'set-able'
        :return:
        """
        return hash((self.date_start, self.date_end, self.type_event))


class AtypicalEventsList:

    def __init__(self):
        self.set_atypical_events = set()

    def add_atypical_event(self, atypical_event):

        if self.is_in_list(atypical_event):
            print('{} is already in the list'.format(atypical_event.name))
        else:
            self.set_atypical_events.add(atypical_event)

    def is_in_list(self, atypical_event):
        """
        Check whereas an event is in the list or not
        :param atypical_event:
        :return:
        """

        bool_is_in_list = False
        if atypical_event in self.set_atypical_events:
            bool_is_in_list = True

        return bool_is_in_list

    def get_events_list(self):
        """
        Gathered all information of the event in a pandas dataframe sorted by 'date_start'
        :return: pandas Dataframe of all the events contained in the list
        """
        # REMARK: Maybe possible to optimize ?
        # - Not using a for loop ?
        # - Storing the ds to avoid reading all events when using the function a second time

        # Gather events in ds
        events_df = pd.DataFrame(columns=['name', 'date_start', 'date_end', 'duration', 'type_event', 'is_atypical'])
        for event in self.set_atypical_events:
            events_df = events_df.append(event.get_info(), ignore_index=True)

        # Sort ds according to date_start
        events_df = events_df.sort_values('date_start')
        events_df = events_df.reset_index(drop=True)

        return events_df

    def get_union(self, other):
        """
        Return union of two AtypicalEventList ACCORDING TO THE STRICT EQUALITY OF EVENTS
        :param other:
        :return:
        """

        union_events_list = AtypicalEventsList()
        union_events_list.set_atypical_events = self.set_atypical_events | other.set_atypical_events

        return union_events_list

    def get_intersection(self, other):
        """
        Return intersection of two AtypicalEventList ACCORDING TO THE STRICT EQUALITY OF EVENTS
        :param other:
        :return:
        """

        intersection_events_list = AtypicalEventsList()
        intersection_events_list.set_atypical_events = self.set_atypical_events & other.set_atypical_events

        return intersection_events_list

    def strict_comparison(self, other):
        """
        STRICTLY compare two lists of AtypicalEventList objects:
            - the current object is the reference
            - 'other' is the object to compare with the reference
        Return an AtypicalEventList object containing the union of the events where:
            - 'is_atypical' attribute is set to 1 for events presents in the reference (True Positive)
            - 'is_atypical' attribute is set to 0 for events not presents in the reference (False Positive)
            - 'is_atypical' attribute is set to -1 for events not presents in other but in reference (False Negative)

        :param other: AtypicalEventList object
        :return: AtypicalEventList object
        """

        compared_object = self.get_union(other)

        for event in compared_object.set_atypical_events:
            if event not in self.get_intersection(other).set_atypical_events:
                # False Negative
                if event in self.set_atypical_events:
                    event.set_isatypical(-1)
                # False Positive
                else:
                    event.set_isatypical(0)
                # True Positive
            else:
                event.set_isatypical(1)

        return compared_object

def get_atypical_events_list(atypical_df, atypical_name):
    """
    From an atypical dataframe return an AtypicalEventsList object
    Only for conso and dayly event for now !

    :param atypical_df:
    :param atypical_name:
    :return:
    """

    mask = (atypical_df.ds.dt.hour == 0) & (atypical_df.ds.dt.minute == 0) & (atypical_df.is_atypical == 1)

    events_list = AtypicalEventsList()

    idx = 0
    for _, row in atypical_df[mask].iterrows():
        name_event = atypical_name + '_' + str(idx)
        date_start = row.ds
        date_end = date_start + datetime.timedelta(hours=23, minutes=45)
        type_event = 1

        event = AtypicalEvent(name_event, date_start, date_end, type_event)

        events_list.add_atypical_event(event)

        idx += 1

    return events_list

def apply_ael_to_df(atypical_df, ael_reference):
    """
    apply 'is_atypical' from ael events to a dataframe containing 'ds' and 'is_atypical' columns

    :param atypical_df:
    :param ael_reference:
    :return:
    """

    atypical_new_df = atypical_df.copy()
    atypical_new_df.is_atypical = np.nan
    atypical_new_df = atypical_new_df.set_index('ds', drop=True)

    for event in ael_reference.set_atypical_events:
        atypical_new_df.loc[event.date_start:event.date_end, 'is_atypical'] = event.is_atypical

    atypical_new_df = atypical_new_df.reset_index()

    return atypical_new_df


def get_confusion_matrix(ael_results, nb_events):

    ael_results_ds = ael_results.get_events_list()

    dict_confusion_matrix = {}
    dict_confusion_matrix['a'] = (ael_results_ds.is_atypical == 1).sum()
    dict_confusion_matrix['b'] = (ael_results_ds.is_atypical == 0).sum()
    dict_confusion_matrix['c'] = (ael_results_ds.is_atypical == -1).sum()
    dict_confusion_matrix['d'] = nb_events - dict_confusion_matrix['a'] - dict_confusion_matrix['b'] - \
                                  dict_confusion_matrix['c']

    return dict_confusion_matrix


def dataframe_daily_to_ael(df):

    ael = AtypicalEventsList()

    for idx, row in df.iterrows():

        date_start = row.ds
        date_end = date_start + datetime.timedelta(hours=23, minutes=45)
        type_event = 1
        name = 'ae_' + str(idx)
        event = AtypicalEvent(name=name, date_start=date_start, date_end=date_end, type_event=type_event)

        ael.add_atypical_event(event)

    return ael


#To Test
if __name__ == '__main__':
    import os
    import pickle

    # Loading holiday Days as atypical events
    path_data_folder = os.path.join("/local/home/antorosi/Documents/Prediction/data")

    holiday_days_csv = os.path.join(path_data_folder, "joursFeries.csv")
    holiday_days_df = pd.read_csv(holiday_days_csv, sep=";")
    holiday_days_df.ds = pd.to_datetime(holiday_days_df.ds)

    hd_event_list = AtypicalEventsList()

    for idx, row in holiday_days_df.iterrows():

        date_start = row.ds
        # for just 2013-2016 period
        if date_start >= datetime.datetime(2013,1,1) and date_start <= datetime.datetime(2017,12,31):

            date_end = date_start + datetime.timedelta(hours=23, minutes=45)
            type_event = 1
            name = 'hd_' + str(idx)
            hd_event = AtypicalEvent(name=name, date_start=date_start, date_end=date_end, type_event=type_event)

            if date_start.weekday()<5:
                hd_event_list.add_atypical_event(hd_event)


    with open(os.path.join(path_data_folder, 'ae_reference_list_wwe_2013-2017' + '.pickle'), 'wb') as f:
        pickle.dump(hd_event_list, f)

    holiday_days_df = holiday_days_df.head(15)
    hd_event_list_2 = AtypicalEventsList()

    for idx, row in holiday_days_df.iterrows():
        date_start = row.ds
        date_end = date_start + datetime.timedelta(hours=23, minutes=45)
        type_event = 1
        name = 'hd_' + str(idx)
        hd_event = AtypicalEvent(name=name, date_start=date_start, date_end=date_end, type_event=type_event)

        hd_event_list_2.add_atypical_event(hd_event)