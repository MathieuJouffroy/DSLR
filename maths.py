import numpy as np

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
    """
    Returns the standard deviation of a distribution. It measure how
    spread out numbers are (the higher the value the more spread out
    the number are, the lower the value the closer they are to the mean).
    """
    mean = mean_(data)
    deviations = 0
    for x in data:
        deviations += (x - mean) ** 2
    variance = deviations / len(data)
    return variance ** 0.5

def min_(data):
    """ Returns the minimum value from a given list of numbers. """
    min_val = data[0]
    for x in data:
        val = x
        if val < min_val:
            min_val = val
    return min_val

def max_(data):
    """ Returns the maximum value from a given list of numbers. """
    max_val = data[0]
    for x in data:
        val = x
        if val > max_val:
            max_val = val
    return max_val

def percentile_(data, percent):
    """
    Computes the q-th quartile of the data. The quartile measures the spread of
    values above and below the mean. A quartile is one of the 3 values, lower quartile,
    median, and upper quartile, that divides the dataset into four equal parts.
    """
    data.sort()
    qth_idx = (len(data) - 1) * percent
    # .floor() to calculate the nearest integer to a decimal number.
    # .ceil() rounds a number down to its nearest integer.
    floor = np.floor(qth_idx)
    ceil = np.ceil(qth_idx)

    if floor == ceil:
        return data[int(qth_idx)]

    d0 = data[int(floor)] * (ceil - qth_idx)
    d1 = data[int(ceil)] * (qth_idx - floor)
    return d0 + d1
