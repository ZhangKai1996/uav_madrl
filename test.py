import numpy as np
import math
import matplotlib.pyplot as plt

beta_list = [2.0, 1.0, 0.1]
x = np.linspace(0.0, 100.0, 100)
for beta in beta_list:
    y = [-math.log(beta * (v+1e-8)) for v in x]
    # y = [math.exp(beta * v) for v in x]
    plt.plot(x, y, label="{:>+4.2f}".format(beta))


plt.legend()
plt.show()


# import threading
# import time
#
#
# # 新线程执行的代码:
# def loop():
#     print('thread %s is running...' % threading.current_thread().name)
#     n = 0
#     while n < 5:
#         n = n + 1
#         print('thread %s >>> %s' % (threading.current_thread().name, n))
#         time.sleep(1)
#     print('thread %s ended.' % threading.current_thread().name)
#
#
# print('thread %s is running...' % threading.current_thread().name)
# t = threading.Thread(target=loop, name='LoopThread')
# t.start()
# t.join()
# print('thread %s ended.' % threading.current_thread().name)
