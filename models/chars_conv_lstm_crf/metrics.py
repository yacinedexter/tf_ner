"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [99.65, 99.56, 99.51, 99.48, 99.40, 99.60, 99.34, 99.59, 99.54, 99.51]
testa = [94.18, 93.99, 94.21, 93.82, 94.04, 93.74, 93.94, 94.43, 94.18, 93.94]
testb = [91.03, 91.31, 90.89, 90.97, 91.10] #91.01, 91.11, 90.58, 91.39, 90.97,  , 

print(np.mean(train), '±', np.std(train), np.max(train), np.min(train))
print(np.mean(testa), '±', np.std(testa), np.max(testa), np.min(testa))
print(np.mean(testb), '±', np.std(testb), np.max(testb), np.min(testb))
