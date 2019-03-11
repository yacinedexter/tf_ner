"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [99.39, 99.34, 99.19, 99.35, 99.36, 99.43, 99.59, 99.26, 99.41, 99.28]
testa = [93.92, 94.21, 94.23, 94.28, 93.96, 93.55, 93.86, 93.81, 94.04, 94.24]
testb = [90.91, 90.09, 91.01, 90.82, 90.43, 90.90, 90.74, 90.61, 90.53, 91.31]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
