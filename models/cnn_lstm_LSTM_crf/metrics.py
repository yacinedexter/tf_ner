"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [99.39, 99.34, 99.19, 99.35, 99.36, 99.43, 99.59, 99.26, 99.41, 99.28]
testa = [93.92, 94.21, 94.23, 94.28, 93.96, 93.55, 93.86, 93.81, 94.04, 94.24]
testb = [90.79, 91.21, 91.25, 90.75, 91.12, 90.58, 90.60, 90.84, 90.88, 90.36]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
