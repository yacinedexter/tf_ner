"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [99.12, 98.71, 98.66, 98.25, 98.68, 98.49, 99.07, 98.25]
testa = [94.33, 94.41, 94.11, 93.88, 94.07, 93.84, 94.22, 94.12]
testb = [90.89, 91.06, 90.23, 90.60, 91.06, 90.71, 90.91, 90.46]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
