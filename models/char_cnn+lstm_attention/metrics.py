"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [0]
testa = [0]
testb = [90.47, 90.46, 90.40, 90.34, 90.62]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
