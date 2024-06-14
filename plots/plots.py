# %%
from matplotlib import pyplot as plt
import numpy as np
# %%
# Relu plot
def reLU(xs):
    relu_x = np.max([xs, np.zeros(xs.shape)], axis=0)
    return relu_x

xmin = -5
xmax = -xmin

ymin = -2
ymax = 5
xs = np.linspace(-5, 5, 100)

plt.plot(xs, reLU(xs), color='black')
plt.axvline(x = 0, color = 'r', label = 'mean outlier score')
ax = plt.gca() # get current axis
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

plt.title("relu function: max(0,x)")
plt.xlabel("x")
plt.ylabel('relu(x)')

plt.show()
# %%
