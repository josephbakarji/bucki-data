import pdb
import json
import os
from IPython.display import display, Math
import numpy as np
import itertools
import random
import pandas as pd


def find_closest_fraction(x, tol=0.01, max_degree=10, return_frac=False):
    """
    This function finds the closest and simplest fraction associated with the input x (up to denominator max_degree)
    """
    xp = np.abs(x)
    x0 = np.floor(xp)
    dx = xp - x0
    if dx < tol:
        if return_frac:
            return int(np.sign(x) * x0), 0, 1
        else:
            return int(np.sign(x) * x0)
    for denominator in range(1, max_degree):
        for numerator in range(1, denominator+1): # Includes 1/1.
            if np.abs(numerator/denominator - dx) <= tol:
                if return_frac:
                    return int(np.sign(x) * x0), numerator, denominator
                else:
                    number = np.sign(x) * (x0 + numerator/denominator)
                    if number % 1 == 0:
                        return int(number)
                    else:
                        return number


def prettify_results(number, names, tol=0.1, max_degree=10):
    
    """
    Turns outputs into Latex and displays it
    
    Parameters
    ----------
    number: dimensionless number powers (the ones you get from learning)
    names: their associated names with same length as number
    tol: tolerance for finding closest simplest fraction
    max_degree: the maximum denominator (of the simplest fraction) to search over.
   
    """

    print(number)
    rounded_nums = []
    for i, xelem in enumerate(number):
        xelem_round = find_closest_fraction(xelem, tol=tol, max_degree=max_degree, return_frac=False)
        if xelem_round is None:
            raise Exception('Tolerance of '+str(tol)+' is too low. Increase value')
        print(names[i],':', xelem_round)
        rounded_nums.append(xelem_round)

    ## Latexify
    rounded_nums_arr = np.array(rounded_nums)
    posidx = np.where(rounded_nums_arr>0)[0]
    negidx = np.where(rounded_nums_arr<0)[0]
    latex_num = ['$\\frac{']
    for i in posidx:
        latex_num.append(names[i])
        if rounded_nums[i] != 1:
            latex_num.append('^{')
            if type(rounded_nums[i]) is int:
                latex_num.append(str(rounded_nums[i]))
            else:
                latex_num.append('{:.3f}'.format(rounded_nums[i]))
            latex_num.append('}')
    latex_num.append('}{')
    for i in negidx:
        latex_num.append(names[i])
        if np.abs(rounded_nums[i]) != 1:
            latex_num.append('^{')
            if type(rounded_nums[i]) is int:
                latex_num.append(str(np.abs(rounded_nums[i])))
            else:
                latex_num.append('{:.3f}'.format(np.abs(rounded_nums[i])))
            latex_num.append('}')
    latex_num.append('}$')
    display(Math(r''.join(latex_num)))


def get_hyperparameter_list(hyperparams):
    def dict_product(dicts):
        return [dict(zip(dicts, x)) for x in itertools.product(*dicts.values())]
    hyperparams_list = dict_product(hyperparams)
    random.shuffle(hyperparams_list)
    return hyperparams_list


def save_results(data, savefile, savedir='./'):
    if os.path.exists(savedir+savefile):
        f = open(savedir+savefile)
        data_list = json.load(f)
        f.close()
    else:
        data_list = []

    data_list.append(data)

    with open(savedir+savefile, 'w') as g:
        json.dump(data_list, g)


def read_results(filename, savedir='./'):
    f = open(savedir+filename)
    data_list = json.load(f)
    f.close()
    return data_list

def read_process_results(filename, savedir='./', filters=None, sortby=None, pickcols=None):
    '''
    filters: dict: {key: val} examples: {'l1_reg': 0.0001}
    sortby: str: 'params.nsamp'
    pickrows: list: ['x_list', 'loss_list']
    '''
    data_list = read_results(filename, savedir)
    pdata = pd.json_normalize(data_list, max_level=1)
    if filters is not None:
        for key, val in filters.items():
            pdata = pdata[(pdata[key] == val)]

    if sortby is not None: 
        pdata = pdata.sort_values(by=[sortby])
    
    if pickcols is not None:
        pdata = pdata[pickcols]

    return pdata

def analyze_data(dataframe, true_value=[-.5, 1, .5, -.5]):

    def get_num_nones(x, eq=None):
        count = 0
        for elem in x:
            if elem == eq:
                count += 1
        return count

    def get_nontrivials(x):
        xn = []
        for xelem in x:
            if xelem is None or xelem == -1:
                pass
            else:
                xn.append(xelem)
        return xn

    def get_nontrivials_loss(x, loss):
        xn = []
        for i, xelem in enumerate(x):
            if xelem is None or xelem == -1:
                pass
            else:
                xn.append(loss[i])
        return xn


    true_value = np.array(true_value)

    dataframe['num_none']   = dataframe.apply(lambda row: get_num_nones(row.x_list, eq=None), axis=1)
    dataframe['num_m1']     = dataframe.apply(lambda row: get_num_nones(row.x_list, eq=-1), axis=1)
    dataframe['x_net']      = dataframe.apply(lambda row: get_nontrivials(row.x_list), axis=1)
    dataframe['loss_net']   = dataframe.apply(lambda row: get_nontrivials_loss(row.x_list, row.loss_list), axis=1)
    dataframe['num_net']    = dataframe.apply(lambda row: len(row.loss_net), axis=1)

    dataframe['argmin_loss']  = dataframe['loss_net'].apply(lambda x: np.argmin(np.array(x)))
    dataframe['x_opt']        = dataframe.apply(lambda row: row.x_net[row.argmin_loss], axis=1)
    dataframe['loss_opt']     = dataframe.apply(lambda row: row.loss_net[row.argmin_loss], axis=1)
    dataframe['x_norm']       = dataframe.apply(lambda row: [[xe/xn[1] for xe in xn] for xn in row.x_net], axis=1)
    dataframe['x_opt_norm']   = dataframe.apply(lambda row: row.x_norm[row.argmin_loss], axis=1)
    dataframe['error_opt']    = dataframe.apply(lambda row: np.linalg.norm(np.array(row.x_opt_norm) - true_value, ord=2), axis=1) 
    dataframe['error_mean']   = dataframe.apply(lambda row: np.mean(np.array([np.linalg.norm(np.array(x) - true_value, ord=2) for x in row.x_norm])), axis=1) 
    dataframe['error_std']    = dataframe.apply(lambda row: np.std(np.array([np.linalg.norm(np.array(x) - true_value, ord=2) for x in row.x_norm])), axis=1) 
    dataframe['error_sol']    = dataframe.apply(lambda row: np.min(np.array([np.linalg.norm(np.array(x) - true_value, ord=2) for x in row.x_norm])), axis=1) 

    return dataframe