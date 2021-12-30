import numpy as np
import matplotlib.pyplot as plt
import pycharm_dark as dk

x = np.linspace(0, 2 * np.pi, 100)
y = np.cos(x)

fig = plt.figure(figsize=(10, 5))
fig.patch.set_facecolor(dk.facecolor)

ax = fig.add_subplot()
ax.set_facecolor(dk.facecolor)

ax.plot(x, y)
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1, 1)

fig.show()
