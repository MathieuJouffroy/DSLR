import numpy as np
import math

def count_(data):
    """ Returns the count of a given list of numbers. """
    return len(data)

def mean_(data):
    """ Returns the mean of a given list of numbers. """
    assert len(data) > 0
    total = 0
    for x in data:
        total += x
    return total / len(data)

def std_(data):
    """ Returns the standard deviation of a distribution. """
    mean = mean_(data)
    total = 0
    for x in data:
        total += (x - mean) ** 2
    return (total / len(data)) ** 0.5

def min_(data):
    """ Returns the minimum value from a given list of numbers. """
    min_value = data[0]
    for x in data:
        val = x
        if val < min_value:
            min_value = val
    return min_value

def max_(data):
    """ Returns the maximum value from a given list of numbers. """
    min_value = data[0]
    for x in data:
        val = x
        if val > min_value:
            min_value = val
    return min_value

def percentile_(data, percent):
    """ Computes the q-th percentile of the data """
    data.sort()
    k = (len(data) - 1) * percent
    floor = np.floor(k)
    ceil = np.ceil(k)

    if floor == ceil:
        return data[int(k)]

    d0 = data[int(floor)] * (ceil - k)
    d1 = data[int(ceil)] * (k - floor)
    return d0 + d1
