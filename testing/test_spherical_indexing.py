# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:30:41 2026

@author: xm904103
"""
import matplotlib.pyplot as plt
import numpy as np

n = [20, 18]

p = [2] * 2

k = 3

# X,Y = np.meshgrid(np.arange(n[0]), np.arange(n[1]))

ix = n[0] - 1

iy = n[1] - 1

xindex = np.zeros([2 * k, 2 * k], dtype=int)
yindex = np.zeros([2 * k, 2 * k], dtype=int)

D = np.ones(n) * 127

for i in range(2 * k):
    ixi = ix + i
    ixi = [ixi, ixi % n[0], ixi % n[0]][p[0]]
    # ixi = (ix + i) % n[0] if p[0] else ix + i
    for j in range(2 * k):
        iyj = iy + j
        ixip = (ixi + n[0] // 2) % n[0] if p[1] == 2 and iyj >= n[1] else ixi
        iyj = [iyj, iyj % n[1], (n[1] - 1 - iyj) % n[1]][p[1]]
        xindex[i, j] = ixip
        yindex[i, j] = iyj
        D[ixip, iyj] = 255

print(xindex)
print(yindex)

plt.imshow(D.T, origin="lower")
plt.show()
