"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [99.46, 99.34, 99.45, 99.56, 99.47, 99.36, 99.53, 99.49, 99.56, 99.21]
testa = [94.56, 94.21, 94.27, 93.62, 94.08, 94.45, 94.10, 94.19, 94.36, 93.93]
testb = [91.09, 90.79, 90.49, 90.65, 90.97, 90.82, 90.98, 91.01, 90.34, 89.98]

print(np.mean(train), np.std(train), np.max(train), np.min(train))
print(np.mean(testa), np.std(testa), np.max(testa), np.min(testa))
print(np.mean(testb), np.std(testb), np.max(testb), np.min(testb))
