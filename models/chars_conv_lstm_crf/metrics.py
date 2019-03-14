"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [99.67, 99.53, 99.55, 99.59, 99.48, 99.45, 99.58, 99.64, 99.59, 99.55]
testa = [94.24, 93.84, 94.01, 94.19, 94.14, 93.95, 93.88, 94.31, 94.03, 94.12]
testb = [90.89, 90.88, 90.77, 90.73, 90.83, 90.93, 90.89, 90.68, 91.24, 90.47]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
