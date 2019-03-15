"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [99.30, 99.23, 99.37, 99.39, 99.50, 99.45, 99.69, 99.52, 99.44, 99.65]
testa = [94.15, 94.04, 94.03, 94.01, 94.15, 93.73, 94.16, 94.34, 93.99, 93.91]
testb = [90.65, 90.19, 90.53, 90.85, 90.61, 90.20, 90.95, 90.90, 90.82, 91.39]

print(np.mean(train), np.std(train), np.max(train), np.min(train))
print(np.mean(testa), np.std(testa), np.max(testa), np.min(testa))
print(np.mean(testb), np.std(testb), np.max(testb), np.min(testb))
