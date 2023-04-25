# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# 2D plot with yellow and green sin/cos curves
xAxis = np.linspace(0, 2*np.pi)
yPlot1 = np.sin(xAxis)
yPlot2 = np.cos(xAxis)
plt.plot(xAxis, yPlot1, color='yellow')
plt.plot(xAxis, yPlot2, color='green')
plt.show()


yAxis = np.linspace(0, 2*np.pi)

x, y = np.meshgrid(xAxis, yAxis)
z = 2*np.sin(x)+np.cos(y)

ax1 = plt.subplot(211)
ax1.imshow(z, cmap='gray')
ax2 = plt.subplot(212)
ax2.imshow(z, cmap='gray')
plt.show()
