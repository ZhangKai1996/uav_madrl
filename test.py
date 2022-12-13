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


from reinforcement_learning.env.util import *

with open('reinforcement_learning/env/record.csv', 'r', newline='') as f:
    for line in f.readlines():
        print(line)
        line = line.strip('\r\n').split(',')
        line = [float(v) for v in line]
        """        
            [px, py, pz, pos.x_val, pos.y_val, pos.z_val, 
             vel.x_val, vel.y_val, vel.z_val, x_z_view, x_y_view]
        """
        print(line)
        print(line[6:9])
        y_z_vel = absolute_bearing((line[7], line[8]))
        x_z_vel = absolute_bearing((line[6], line[8]))
        x_y_vel = absolute_bearing((line[6], line[7]))
        print('x v:', y_z_vel)
        print('y v:', x_z_vel)
        print('z v:', x_y_vel)
        print()

        p1, p0 = (line[0], line[1]), (line[3], line[4])
        print(p0, p1)
        print('\tNED z:', relative_bearing(p0=p0, p1=p1, reference=x_y_vel))
        print('\tNED z:', relative_bearing(p0=p0, p1=p1))
        print('\tNED z o:', absolute_bearing(p0))
        print('\tNED z p:', absolute_bearing(p1))
        print(relative_bearing(p0=p0, p1=p1, reference=x_y_vel) == line[-1])
        print()

        p1, p0 = (line[0], line[2]), (line[3], line[5])
        print(p0, p1)
        print('\tNED y:', relative_bearing(p0=p0, p1=p1))
        print('\tNED y o:', absolute_bearing(p0))
        print('\tNED y p:', absolute_bearing(p1))
        print()

        p1, p0 = (line[1], line[2]), (line[4], line[5])
        print(p0, p1)
        print('\tNED x:', relative_bearing(p0=p0, p1=p1))
        print('\tNED x o:', absolute_bearing(p0))
        print('\tNED x p:', absolute_bearing(p1))
        print()
