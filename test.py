import math

import numpy as np
import matplotlib.pyplot as plt


# x = np.linspace(0.0, 100.0, 100)
#
# for beta in np.linspace(.0, .1, 11):
#     if beta <= 0.0:
#         continue
#
#     y = [math.log10(beta * (v+1e-8)) for v in x]
#     # y = [math.exp(beta * v) for v in x]
#     plt.plot(x, y, label="{:>+4.2f}".format(beta))
#
#
# plt.legend()
# plt.show()


from reinforcement_learning.env.drone_env import bearing

print(bearing(p1=(1.0, 1.0)))
print(bearing(p1=(-1.0, 1.0)))
print(bearing(p1=(-1.0, -1.0)))
print(bearing(p1=(1.0, -1.0)))

print(math.cos(math.radians(90)))
