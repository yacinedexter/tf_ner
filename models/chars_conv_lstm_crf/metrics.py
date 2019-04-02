"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [99.65, 99.56, 99.51, 99.48, 99.40, 99.60, 99.34, 99.59, 99.54, 99.51]
testa = [94.18, 93.99, 94.21, 93.82, 94.04, 93.74, 93.94, 94.43, 94.18, 93.94]
testb = [90.65, 90.85, 90.81, 90.71, 90.12, 90.77, 91.15, 90.89, 90.89, 90.61] #

print(np.mean(train), '±', np.std(train), np.max(train), np.min(train))
print(np.mean(testa), '±', np.std(testa), np.max(testa), np.min(testa))
print(np.mean(testb), '±', np.std(testb), np.max(testb), np.min(testb))
