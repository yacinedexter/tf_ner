"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [98.33, 98.68, 98.57, 98.74, 98.94, 98.34, 98.63, 98.63, 98.59, 98.98]
testa = [94.14, 94.41, 94.01, 93.98, 94.30, 93.92, 94.07, 94.07, 94.04, 94.16]
testb = [90.33, 90.73, 90.66, 90.49, 90.84, 90.41, 90.73, 90.73, 90.58, 90.73]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
