import setup_path

import copy

import torch as th
import numpy as np
import cv2
from torchviz import make_dot

standard_width = 1600
standard_height = 900


def net_visual(dim_input, net, d_type=th.FloatTensor, **kwargs):
    """
    支持MLP、CNN和RNN三种网络的可视化。
    """
    print(dim_input, kwargs['filename'])
    xs = [th.randn(*dim).type(d_type).requires_grad_(True) for dim in dim_input]  # 定义一个网络的输入值
    y = net(*xs)  # 获取网络的预测值
    print(y.shape)
    net_vis = make_dot(y, params=dict(list(net.named_parameters()) + [('x', x) for x in xs]))
    net_vis.render(**kwargs)     # 生成文件
    print('Save to ' + kwargs['directory'] + '{}.{}'.format(kwargs['filename'], kwargs['format']))


def border_func(x, min_v=0.0, max_v=1.0, d_type=float):
    """
    与最大值取较小者，与最小值取较大者，返回值的类型取决于d_type。
    """
    print(x, min_v, max_v, d_type)
    return d_type(min(max(x, min_v), max_v))


# todo: 1) 改成动画的形式；2) 加入CNN和RNN；
class NetLooker:
    """
    The visualization of MLP layer.
    """
    def __init__(self, net, name='net', is_look=True, root=None, width=16000, height=9000, channel=3, ratio=0.5,
                 **kwargs):
        self.name = name
        self.net = net
        self.vars = net.state_dict()

        self.width = width
        self.height = height
        self.channel = channel
        self.ratio = ratio
        self.root = root

        self.is_look = is_look  # 临时参数，因为只支持可视化MLP，CNN和RNN尚不支持，功能完善之后删除会此参数。
        if is_look:
            self.__look_weights_and_biases(**kwargs)

    def __look_weights_and_biases(self, look_weight=False, look_bias=True, scale=10):
        # 创建一个的黑色画布，RGB(0,0,0)即黑色
        image = np.ones((self.height, self.width, self.channel), np.uint8) * 255
        length = len(self.vars)
        num_layers = length // 2 + 1
        c_start = int(self.height * 0.05)  # 最左侧节点圆心与左侧边界的距离
        r_interval = (self.height - 2 * c_start) // (num_layers - 1)  # 相邻层节点圆心的距离
        c_interval = int(self.width // max([var.shape[-1] for var in self.vars.values()]))  # 同层相邻节点圆心的距离
        self.radius = int(c_interval * 0.4)

        var_list, link_metrix, count = [], [], 0
        for i, (var_name, var) in enumerate(self.vars.items()):
            var_list.append(var)
            if i % 2 != 0 and i != length - 1:
                continue

            dim_nodes = var.shape[-1]
            if count == 0:
                mode = 'input'
                string = "{}({})".format(mode, dim_nodes)
            elif count == num_layers - 1:
                mode = 'output'
                string = "{}({},{})".format(mode, dim_nodes, 'Tanh')
            else:
                mode = 'hidden'
                string = "{}({},{})".format(mode, dim_nodes, 'Relu')

            image = cv2.putText(image, string,
                                (self.width // 2, c_start + count * r_interval + 10 * self.radius),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                self.width / standard_width * self.ratio, (255.0, 0, 0),
                                thickness=self.radius // 2)
            # 最上层节点圆心与上层边界的距离
            r_start = int(self.width // 2 - (dim_nodes - 1) / 2 * c_interval)

            coord_list = []
            for j in range(dim_nodes):
                coord = (r_start + j * c_interval, c_start + count * r_interval)
                coord_list.append(coord)
            link_metrix.append(coord_list)

            count += 1

        for i in range(len(var_list) // 2):
            weights, biases = var_list[i * 2], var_list[i * 2 + 1]
            # print(weights.shape, len(link_metrix[i]), len(link_metrix[i + 1]))

            for j, coord1 in enumerate(link_metrix[i]):
                for k, coord2 in enumerate(link_metrix[i + 1]):
                    # bias的颜色（越大越长，负下正上）
                    if j == len(link_metrix[i + 1]) - 1 and look_bias:
                        c_bias = 255 if abs(biases[k] * scale) >= 1.0 else 150
                        image = cv2.line(image,
                                         coord2,
                                         (coord2[0], coord2[1] - border_func(biases[k] * scale, d_type=int) * 5 * self.radius),
                                         (0, 0, c_bias),
                                         thickness=self.radius // 2)
                    # weight连线的颜色（带符号的权重越大，越接近黑色）
                    if look_weight:
                        c_weight = (1.0 - border_func(abs(weights[k, j]), d_type=float)) * 255.0
                        image = cv2.line(image,
                                         (coord1[0], coord1[1] + self.radius), (coord2[0], coord2[1] - self.radius),
                                         (c_weight, c_weight, c_weight))
        self.link_matrix = link_metrix
        self.image = image

    def __look_layer(self, image, values, matrix):
        assert len(matrix) == len(values)
        for i, v in enumerate(values):
            thick = self.radius // 2 if v < 0 else -1
            c_node = (1 - border_func(abs(v), d_type=float)) * 255.0
            image = cv2.circle(image, matrix[i], self.radius, (c_node, c_node, c_node), thickness=thick)
        return image

    def look(self, inputs, name='0', suffix='png'):
        if not self.is_look:
            return

        # 底图
        image = copy.deepcopy(self.image)
        # 得到网络每一层的输出，并画在底图上
        outputs = self.net.forward_(th.from_numpy(inputs).float())
        for i, out in enumerate(outputs):
            image = self.__look_layer(image, out[0, :], self.link_matrix[i])
        # 调整图片的分辨率
        image = cv2.resize(image, (0, 0), fx=self.ratio, fy=self.ratio)
        # 显示图片
        cv2.imshow(self.name, image)
        # 按q保存图片
        if self.root is None:
            save_path = '{}_{}.{}'.format(self.name, name, suffix)
        else:
            save_path = '{}/{}_{}.{}'.format(self.root, self.name, name, suffix)
        if cv2.waitKey(0) == 113:
            cv2.imwrite(save_path, image)

    def close(self):
        cv2.waitKey(1) & 0xFF
        cv2.destroyAllWindows()
