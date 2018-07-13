# The notations and the names of the indicators are the same of the book 'Forecast Verification'
# from Ian T. Jolliffe and DavidB.Stephenson

# Same for the names of the confusion matrix elements. From a machine learning perspective:
# - a corresponds to the True Positive (TP)
# - b corresponds to the False Positive (FP)
# - c corresponds to the False Negative (FN)
# - d corresponds to the True Negative (TN)

from aed.atypical_event import get_confusion_matrix
from aed.multi_period import get_aed_results


def get_h(dict_confusion_matrix):
    """
    Hit rate
    :param dict_confusion_matrix:
    :return:
    """

    h = (dict_confusion_matrix['a'])/(dict_confusion_matrix['a']+dict_confusion_matrix['c'])

    return h


def get_f(dict_confusion_matrix):
    """
    False alarm rate
    :param dict_confusion_matrix:
    :return:
    """
    f = (dict_confusion_matrix['b'])/(dict_confusion_matrix['b']+dict_confusion_matrix['d'])

    return f


def get_far(dict_confusion_matrix):
    """
    False alarm ratio
    :param dict_confusion_matrix:
    :return:
    """

    if (dict_confusion_matrix['a'] != 0) or (dict_confusion_matrix['b'] != 0):
        far = (dict_confusion_matrix['b'])/(dict_confusion_matrix['a']+dict_confusion_matrix['b'])
    else:
        far = 0

    return far

def get_ppv(dict_confusion_matrix):
    """
    False alarm ratio
    :param dict_confusion_matrix:
    :return:
    """

    if (dict_confusion_matrix['a'] != 0) or (dict_confusion_matrix['b'] != 0):
        ppv = (dict_confusion_matrix['a'])/(dict_confusion_matrix['a']+dict_confusion_matrix['b'])
    else:
        ppv = 0

    return ppv

def get_pc(dict_confusion_matrix):
    """
    Proportion correct
    :param dict_confusion_matrix:
    :return:
    """
    n = dict_confusion_matrix['a'] + dict_confusion_matrix['b'] + dict_confusion_matrix['c'] + dict_confusion_matrix['d']
    pc = (dict_confusion_matrix['a']+dict_confusion_matrix['d'])/n

    return pc


def get_csi(dict_confusion_matrix):
    """
    Critical success index
    :param dict_confusion_matrix:
    :return:
    """

    csi = dict_confusion_matrix['a']/(dict_confusion_matrix['a']+dict_confusion_matrix['b']+dict_confusion_matrix['c'])

    return csi


def get_gss(dict_confusion_matrix):
    """
    Gilberts's skill score
    :param dict_confusion_matrix:
    :return:
    """

    n = dict_confusion_matrix['a']+dict_confusion_matrix['b']+dict_confusion_matrix['c']+dict_confusion_matrix['d']
    ar = (dict_confusion_matrix['a']+dict_confusion_matrix['b'])*(dict_confusion_matrix['a']+dict_confusion_matrix['c'])/n

    gss = (dict_confusion_matrix['a']-ar)/(dict_confusion_matrix['a'] - ar + dict_confusion_matrix['b'] + dict_confusion_matrix['c'])

    return gss


def get_pss(dict_confusion_matrix):
    """
    Peirce's skill score.
    Also known as True skill statistic (TSS)
    :param dict_confusion_matrix:
    :return:
    """

    pss = get_h(dict_confusion_matrix) - get_f(dict_confusion_matrix)

    return pss


def get_hss(dict_confusion_matrix):
    """
    Heidke skill score
    :param dict_confusion_matrix:
    :return:
    """

    n = dict_confusion_matrix['a'] + dict_confusion_matrix['b'] + dict_confusion_matrix['c'] + dict_confusion_matrix['d']

    pc = get_pc(dict_confusion_matrix)

    e = (dict_confusion_matrix['a']+dict_confusion_matrix['c'])*(dict_confusion_matrix['a']+dict_confusion_matrix['b'])/(n*n) \
      + (dict_confusion_matrix['b']+dict_confusion_matrix['d'])*(dict_confusion_matrix['c']+dict_confusion_matrix['d'])/(n*n)

    hss = (pc-e)/(1-e)

    return hss

def get_f1(dict_confusion_matrix):
    """
    F1 score
    :param dict_confusion_matrix:
    :return:
    """

    f1 = 2*dict_confusion_matrix['a'] / (
                2*dict_confusion_matrix['a'] + dict_confusion_matrix['b'] + dict_confusion_matrix['c'])

    return f1

def get_all_scores(dict_connfusion_matrix):
    """
    Get all scores
    :param dict_connfusion_matrix:
    :return:
    """

    dict_scores = {}
    dict_scores['h'] = get_h(dict_connfusion_matrix)
    dict_scores['f'] = get_f(dict_connfusion_matrix)
    dict_scores['far'] = get_far(dict_connfusion_matrix)
    dict_scores['pc'] = get_pc(dict_connfusion_matrix)
    dict_scores['csi'] = get_csi(dict_connfusion_matrix)
    dict_scores['gss'] = get_gss(dict_connfusion_matrix)
    dict_scores['pss'] = get_pss(dict_connfusion_matrix)
    dict_scores['hss'] = get_hss(dict_connfusion_matrix)
    dict_scores['f1'] = get_f1(dict_connfusion_matrix)
    dict_scores['ppv'] = get_ppv(dict_connfusion_matrix)

    return dict_scores

def get_all_scores_multi_t(list_threshold, prediction_results, ael_reference, nb_events):

    # initialization


    threshold = list_threshold[0]
    aed_results, ael_full_model = get_aed_results(prediction_results, threshold)
    ael_results = ael_reference.strict_comparison(ael_full_model)
    dict_confusion_matrix = get_confusion_matrix(ael_results, nb_events)
    dict_result = get_all_scores(dict_confusion_matrix)

    dict_results = {}
    for key, value in dict_result.items():
        dict_results[key] = [value]

    # loop
    for threshold in list_threshold[1:]:
        aed_results, ael_full_model = get_aed_results(prediction_results, threshold)
        ael_results = ael_reference.strict_comparison(ael_full_model)
        dict_confusion_matrix = get_confusion_matrix(ael_results, nb_events)
        dict_result = get_all_scores(dict_confusion_matrix)

        for key, value in dict_result.items():
            dict_results[key].append(value)

    dict_results['threshold'] = list_threshold

    return dict_results