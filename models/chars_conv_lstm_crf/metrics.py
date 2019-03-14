"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [99.02, 99.13, 99.24, 99.07, 99.28, 99.16, 99.05, 99.29, 99.10, 99.25]
testa = [94.21, 93.91, 94.13, 93.83, 93.93, 93.87, 94.33, 93.80, 93.90, 93.72]
testb = [90.49, 91.06, 90.59, 90.86, 91.03, 90.89, 90.78, 91.06, 90.91, 90.89]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
