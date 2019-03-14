"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [98.33, 98.68, 98.57, 99.02, 98.94, 98.34, 98.22, 98.63, 98.59, 98.98]
testa = [94.14, 94.41, 94.01, 94.24, 94.30, 93.92, 94.16, 94.07, 94.04, 94.16]
testb = [91.01, 90.39, 90.83, 91.04, 91.30, 90.73, 90.51, 91.03, 90.75, 90.48]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
