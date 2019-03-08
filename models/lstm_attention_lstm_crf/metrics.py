"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [99.12, 98.71, 98.66, 98.25, 98.68, 98.49, 99.07, 98.25, 98.63, 98.28]
testa = [94.33, 94.41, 94.11, 93.88, 94.07, 93.84, 94.22, 94.12, 94.36, 93.94]
testb = [90.44, 90.52, 90.74, 90.64, 90.94, 90.48, 90.68, 90.74, 90.98, 90.93, 90.76]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
