import pdb
from IPython.display import display, Math
import numpy as np



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
