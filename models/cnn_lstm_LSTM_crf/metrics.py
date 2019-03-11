"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [98.90, 98.85, 99.17, 98.95, 99.15, 98.79, 98.94, 99.04, 99.03, 98.94]
testa = [94.11, 94.19, 93.92, 94.20, 94.13, 94.11, 94.31, 94.16, 93.84, 94.17]
testb = [90.59, 90.62, 91.07, 90.32, 90.97, 90.18, 90.50, 90.49, 90.44, 90.60]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
