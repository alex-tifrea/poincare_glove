from numpy import inf, nan


def is_number(x):
    if x not in [inf, -inf, nan]:
        return True
    else:
        return False
