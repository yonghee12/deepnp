from collections import namedtuple


class GradTuples:
    fc = namedtuple('FCLayer', ['W', 'x', 'bias'])
