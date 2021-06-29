import math 

def dot_(v, w):
    assert len(v) == len(w)
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v):
    return dot_(v, v)

def count_(dataset):
    return len(dataset)

def mean_(dataset):
    assert len(dataset) > 0
    return sum(dataset) / count_(dataset)

def distance(dataset):
    mean = mean_(dataset)
    return [x - mean for x in dataset]

def variance_(dataset):
    assert len(dataset) >= 2
    n = len(dataset)
    deviations = distance(dataset)
    return sum_of_squares(deviations) / (n - 1)

def std_(dataset) :
    return math.sqrt(variance_(dataset))

def min_(dataset):
    min = dataset[0]
    for data in dataset:
        if data < min:
            min = data
    return min

def max_(dataset):
    max = dataset[0]
    for data in dataset:
        if data > max:
            max = data
    return max