"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [99.45, 99.57, 99.64, 99.61, 99.56, 99.49, 99.49, 99.52, 99.52, 99.35]
testa = [94.11, 94.17, 93.98, 94.10, 94.12, 94.07, 94.30, 93.99, 93.87, 93.98]
testb = [90.92, 90.98, 91.18, 90.22, 91.13, 90.34, 90.85, 90.90, 91.29, 90.73]

print(np.mean(train), '±', np.std(train), np.max(train), np.min(train))
print(np.mean(testa), '±', np.std(testa), np.max(testa), np.min(testa))
print(np.mean(testb), '±', np.std(testb), np.max(testb), np.min(testb))
