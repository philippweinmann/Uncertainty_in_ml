# %%
from matplotlib import pyplot as plt
import numpy as np
# %%
# Relu plot
def reLU(xs, alpha = 0):
    relu_x = np.max([xs, alpha * xs], axis=0)
    return relu_x

xmin = -5
xmax = -xmin

ymin = -2
ymax = 5
xs = np.linspace(-5, 5, 100)

plt.figure()
plt.plot(xs, reLU(xs), color='black')
plt.axvline(x = 0, color = 'r')
# plt.axhline(y = 0, linestyle="--", color = 'r')
ax = plt.gca() # get current axis
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

plt.title("relu function: max(0,x)")
plt.xlabel("x")
plt.ylabel('relu(x)')

plt.show()
# %%
# plotting leaky relu function

plt.figure()
plt.plot(xs, reLU(xs, alpha=0.1), color='black')
plt.axvline(x = 0, color = 'r')
plt.axhline(y = 0, linestyle="--", color = 'r')
ax = plt.gca() # get current axis
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

plt.title("leaky relu function: f(x) = max(alpha * x,x)")
plt.xlabel("x")
plt.ylabel('leaky_relu(x)')

plt.show()
# %%
