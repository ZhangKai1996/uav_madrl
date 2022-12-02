from copy import deepcopy

import numpy as np
import torch as th
from torch.optim import Adam
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .memory import ReplayMemory, Experience
from .network import Critic, Actor
from .misc import get_folder, soft_update, FloatTensor, ByteTensor
from .visual import NetLooker, net_visual


class Trainer:
    def __init__(self, n_agents, dim_obs, dim_act, args, folder=None):
        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.var = 1.0

        self.memory = ReplayMemory(args.memory_length)

        self.actors = [Actor(dim_obs, dim_act) for _ in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs, dim_act) for _ in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)
        self.critic_optimizer = [Adam(x.parameters(), lr=args.c_lr) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(), lr=args.a_lr) for x in self.actors]

        self.c_losses, self.a_losses = [], []
        self.__addiction(n_agents, dim_obs, dim_act, folder)

    def __addiction(self, n_agents, dim_obs, dim_act, folder):
        self.writer = None
        self.actor_looker = None
        self.critic_looker = None

        if folder is None:
            return

        # 数据记录（计算图、logs和网络参数）的保存文件路径
        self.path = get_folder(folder,
                               has_graph=True,
                               has_log=True,
                               has_model=True,
                               allow_exist=True)
        if self.path['log_path'] is not None:
            self.writer = SummaryWriter(self.path['log_path'])
        if self.path['graph_path'] is not None:
            print('Draw the net of Actor and Critic!')
            net_visual([(1, ) + dim_obs],
                       self.actors[0],
                       filename='actor',
                       directory=self.path['graph_path'],
                       format='png',
                       cleanup=True)
            self.actor_looker = NetLooker(net=self.actors[0],
                                          name='actor',
                                          is_look=False,
                                          root=self.path['graph_path'])
            net_visual([(n_agents, 1,) + dim_obs, (n_agents, 1, dim_act)],
                       self.critics[0],
                       filename='critic',
                       directory=self.path['graph_path'],
                       format='png',
                       cleanup=True)
            self.critic_looker = NetLooker(net=self.critics[0],
                                           name='critic',
                                           is_look=False)

    def add_experience(self, obs_n, act_n, next_obs_n, rew_n, done_n):
        self.memory.push(obs_n, act_n, next_obs_n, rew_n, done_n)

    def act(self, obs_n, var_decay=False):
        n_actions = self.n_actions
        obs_n = th.from_numpy(obs_n).type(FloatTensor)
        act_n = th.zeros(self.n_agents, n_actions)

        for i in range(self.n_agents):
            sb = obs_n[i, :].detach()
            act = self.actors[i](sb.unsqueeze(0)).squeeze()
            act += th.from_numpy(np.random.randn(n_actions) * self.var).type(FloatTensor)
            act = th.clamp(act, -1.0, 1.0)
            act_n[i, :] = act
            if var_decay and self.var > 0.05:
                self.var *= 0.999998
        return act_n.data.cpu().numpy()

    def update(self, step):
        c_loss, a_loss = [], []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))

            non_final_mask = ByteTensor(list(map(lambda s: s is not None, batch.next_states)))
            state_batch = th.stack(batch.states).type(FloatTensor)  # state_batch: batch_size x n_agents x dim_obs
            action_batch = th.stack(batch.actions).type(FloatTensor)
            reward_batch = th.stack(batch.rewards).type(FloatTensor)
            non_final_next_states = th.stack(
                [s for s in batch.next_states if s is not None]
            ).type(FloatTensor)  # : (batch_size_non_final) x n_agents x dim_obs

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            current_q = self.critics[agent](whole_state, whole_action)
            non_final_next_actions = [self.actors_target[i](non_final_next_states[:, i, :])
                                      for i in range(self.n_agents)]
            non_final_next_actions = th.stack(non_final_next_actions)
            non_final_next_actions = (non_final_next_actions.transpose(0, 1).contiguous())
            target_q = th.zeros(self.batch_size).type(FloatTensor)
            target_q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(-1,
                                            self.n_agents * self.n_actions)
            ).squeeze()
            # scale_reward: to scale reward in Q functions
            target_q = (target_q.unsqueeze(1) * self.gamma) + (reward_batch[:, agent].unsqueeze(1) * 0.01)
            loss_q = nn.MSELoss()(current_q, target_q.detach())
            loss_q.backward()
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            ac = action_batch.clone()
            ac[:, agent, :] = self.actors[agent](state_batch[:, agent, :])
            whole_action = ac.view(self.batch_size, -1)
            loss_p = -self.critics[agent](whole_state, whole_action).mean()
            loss_p.backward()
            self.actor_optimizer[agent].step()

            c_loss.append(loss_q.item())
            a_loss.append(loss_p.item())

        self.c_losses.append(c_loss)
        self.a_losses.append(a_loss)

        if step % 100 == 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)
            # Record and visual the loss value of Actor and Critic
            self.writer.add_scalars('critic_loss',
                                    {'agent_{}'.format(i+1): v for i, v in enumerate(np.mean(self.c_losses, axis=0))},
                                    step)
            self.writer.add_scalars('actor_loss',
                                    {'agent_{}'.format(i+1): v for i, v in enumerate(np.mean(self.a_losses, axis=0))},
                                    step)
            self.c_losses, self.a_losses = [], []

    def load_model(self, load_path=None):
        if load_path is None:
            load_path = self.path['model_path']

        if load_path is not None:
            for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
                actor_state_dict = th.load(load_path + 'actor_{}.pth'.format(i)).state_dict()
                critic_state_dict = th.load(load_path + 'critic_{}.pth'.format(i)).state_dict()

                actor.load_state_dict(actor_state_dict)
                critic.load_state_dict(critic_state_dict)
                self.actors_target[i] = deepcopy(actor)
                self.critics_target[i] = deepcopy(critic)
        else:
            print('Load path is empty!')

    def save_model(self, save_path=None):
        if save_path is None:
            save_path = self.path['model_path']

        if save_path is not None:
            for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
                th.save(actor, save_path + 'actor_{}.pth'.format(i))
                th.save(critic, save_path + 'critic_{}.pth'.format(i))
        else:
            print('Save path is empty!')

    def scalars(self, key, value, episode):
        self.writer.add_scalars(key, value, episode)

    def scalar(self, key, value, episode):
        self.writer.add_scalar(key, value, episode)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.actor_looker is not None:
            self.actor_looker.close()
        if self.critic_looker is not None:
            self.critic_looker.close()
