import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_agent, dim_obs, dim_act):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.conv1 = nn.Conv2d(dim_obs[0], 4, kernel_size=9, padding=2)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=7)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=5)
        self.conv4 = nn.Conv2d(8, 8, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.FC = nn.Linear(3136, 128)

        self.FC1 = nn.Linear((128 + dim_act) * n_agent, 256)
        self.FC2 = nn.Linear(256, 128)
        self.FC3 = nn.Linear(128, 1)

    def forward(self, obs_n, act_n):
        batch_size = obs_n.size(0)
        combined = []
        for i in range(self.n_agent):
            out = F.relu(self.mp(self.conv1(obs_n[:, i])))
            out = F.relu(self.mp(self.conv2(out)))
            out = F.relu(self.mp(self.conv3(out)))
            out = F.relu(self.conv4(out))
            out = out.view(batch_size, -1)
            out = F.relu(self.FC(out))
            combined.append(th.cat([out, act_n[:, i]], 1))
        combined = th.cat(combined, 1)
        out = F.relu(self.FC1(combined))
        out = F.relu(self.FC2(out))
        out = self.FC3(out)
        return out


class Actor(nn.Module):
    def __init__(self, dim_obs, dim_act):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(dim_obs[0], 4, kernel_size=9, padding=2)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=7)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=5)
        self.conv4 = nn.Conv2d(8, 8, kernel_size=5)
        self.mp = nn.MaxPool2d(2)

        self.FC1 = nn.Linear(3136, 512)
        self.FC2 = nn.Linear(512, 128)
        self.FC3 = nn.Linear(128, dim_act)

    def forward(self, obs_n):
        """
        obs_n.shape = (batch_size, channel, width, height)
        """
        in_size = obs_n.size(0)
        out = F.relu(self.mp(self.conv1(obs_n)))
        out = F.relu(self.mp(self.conv2(out)))
        out = F.relu(self.mp(self.conv3(out)))
        out = F.relu(self.conv4(out))
        out = out.view(in_size, -1)
        out = F.relu(self.FC1(out))
        out = F.relu(self.FC2(out))
        out = th.tanh(self.FC3(out))
        return out
