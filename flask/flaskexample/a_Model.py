import numpy as np
from recommender import CosSimilarityRecommender

def ModelIt(fromUser  = 'Default', topics=[], user = {}, csr = None, mn = 0, sig = 1):
    lst = []
    for k in user:
        if user[k] == 1:
            lst.append(k)
    vec = np.array([t in lst for t in topics]).astype(int)
    pred = csr.predict((vec - mn) / sig) * sig + mn

    ranks = np.argsort(pred[0])[::-1]

    result = {}

    i = 0
    for r in ranks:
        t = topics[r]
        if t in user.keys():
            continue
        result[topics[r]] = pred[0, r]
        i += 1
        if i > 2:
            break;

    print('Predictions:')
    print(result)

    if fromUser != 'Default':
        return result
    else:
        return 'check your input'
