"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [99.86, 99.82, 99.91, 99.89, 99.81, 99.86, 99.87, 99.80, 99.83, 99.92]
testa = [94.17, 94.30, 94.18, 94.37, 94.18, 94.09, 94.17, 94.25, 94.09, 93.91]
testb = [90.82, 90.91, 91.11, 91.11, 90.96] # , ,

print(np.mean(train), '±', np.std(train), np.max(train), np.min(train))
print(np.mean(testa), '±', np.std(testa), np.max(testa), np.min(testa))
print(np.mean(testb), '±', np.std(testb), np.max(testb), np.min(testb))
