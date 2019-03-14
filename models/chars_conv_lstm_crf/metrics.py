"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [99.61, 99.51, 99.29, 99.54, 99.48, 99.47, 99.65, 99.36, 99.46, 99.37]
testa = [94.25, 94.15, 94.06, 94.28, 94.36, 94.48, 94.21, 94.16, 94.12, 94.17]
testb = [90.49, 91.06, 90.59, 90.86, 91.03, 90.89, 90.78, 91.06, 90.91, 90.89]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
