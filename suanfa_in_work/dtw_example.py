from dtw import dtw
import numpy as np

x = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
y = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0]).reshape(-1, 1)

dist, cost, acc, path = dtw(x, y, dist=lambda x, y: abs(x - y))

print()