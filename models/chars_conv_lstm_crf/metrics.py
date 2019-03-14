"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [9]
testa = [9]
testb = [91.30, 90.53, 90.61, 90.74, 90.88, 90.09, 91.01, 90.90, 90.43, 90.91]

print(np.mean(train), np.std(train), np.max(train), np.min(train))
print(np.mean(testa), np.std(testa), np.max(testa), np.min(testa))
print(np.mean(testb), np.std(testb), np.max(testb), np.min(testb))
