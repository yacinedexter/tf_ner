"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [0]
testa = [0]
testb = [85.39, 85.17, 85.19, 85.05, 85.37]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
