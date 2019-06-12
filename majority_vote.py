import numpy as np

def majority_vote(l):
    sum = 0
    for array in l:
        sum += array

    final_prediction = []
    for x in sum:
        if x >= len(l)/2:
            final_prediction.append(1)
        else:
            final_prediction.append(0)

    return np.array(final_prediction)
