"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [99.41, 99.28, 99.26, 99.19, 99.43, 99.36, 99.59, 99.35, 99.34, 99.39]
testa = [93.81, 94.24, 93.96, 94.04, 93.86, 94.21, 93.55, 94.23, 94.28, 93.92]
testb = [91.09, 91.06, 90.76, 90.43, 90.46, 90.77, 90.77, 90.25, 90.83, 90.85]

print(np.mean(train), np.std(train), np.max(train), np.min(train))
print(np.mean(testa), np.std(testa), np.max(testa), np.min(testa))
print(np.mean(testb), np.std(testb), np.max(testb), np.min(testb))
