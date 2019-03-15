"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [99.56, 99.42, 99.51, 99.62, 99.52, 99.48, 99.62, 99.53, 99.56, 99.44]
testa = [93.98, 94.45, 94.20, 94.30, 94.14, 94.03, 94.38, 93.88, 94.03, 94.36]
testb = [90.65, 90.19, 90.53, 90.85, 90.61, 90.20, 90.95, 90.90, 90.82, 91.39]

print(np.mean(train), np.std(train), np.max(train), np.min(train))
print(np.mean(testa), np.std(testa), np.max(testa), np.min(testa))
print(np.mean(testb), np.std(testb), np.max(testb), np.min(testb))
