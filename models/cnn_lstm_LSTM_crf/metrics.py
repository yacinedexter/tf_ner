"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [99.42, 99.34, 99.39, 99.31, 99.46, 99.54, 99.41, 99.66, 99.50, 99.27]
testa = [93.90, 94.39, 94.43, 94.35, 94.43, 93.67, 94.17, 94.23, 94.03, 93.91]
testb = [90.79, 91.21, 91.25, 90.75, 91.12, 90.58, 90.60, 90.84, 90.88, 90.36]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
