# uav_madrl

## 版本管理

v1.0
  - **初始版本**；

v2.0
  - **Algorithm：** 改为Pytorch的MADDPG算法；
  - **远程连接：** PythonClient和UnrealProject在不同的电脑上；
  - **初步调试：** 可以运行；

v2.1.5
  - **Reward：** 模仿MADDPG论文所使用的设置，调整我们的； 
  - **过程调试：** tensor的size和CNN网络对接；
  - **功能完善：** 对observation增加雷达图(Attitude Director Indicator, ADI)；
  - **训练加速：** GPU cuda加速（v2.1.2）；
  - **多个线程：** Thread（与环境交互和模型更新过程分开），Multiprocessing（模型更新时每个Agent占一个thread）（v2.1.3）； 
  - **再次调试：** 无GPU条件下，训练1240步无报错；
  - **训练加速：** 环境的渲染速度由x1.0调整到x8.0，单步训练时间由3.5秒缩短为0.9秒（v2.1.4）；
  - **内存调整：** 之前trainer中的memory占用的是GPU的显存，现在改为内存（v2.1.5）；
  - **统计变量：** 在统计每一定回合数以步（step）计算的平均Reward的基础之上，增加了以episode计算的平均Reward；
