from multiprocessing import Queue, Process
from copy import deepcopy

import numpy as np
import torch as th
from torch.optim import Adam
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .memory import ReplayMemory, Experience
from .network import Critic, Actor
from .misc import get_folder, soft_update, FloatTensor
from .visual import NetLooker, net_visual


def update_single_agent(agent, queue, n_agents,
                        actor, actors_target, critic, critic_target, a_optimizer, c_optimizer, mse_loss,
                        tau, gamma, memory, batch_size, update_target):
    transitions = memory.sample(batch_size)
    batch = Experience(*zip(*transitions))

    state_batch = th.stack(batch.states).type(FloatTensor)
    action_batch = th.stack(batch.actions).type(FloatTensor)
    next_states_batch = th.stack(batch.next_states).type(FloatTensor)
    reward_batch = th.stack(batch.rewards).type(FloatTensor).unsqueeze(dim=-1)
    done_batch = th.stack(batch.dones).type(FloatTensor).unsqueeze(dim=-1)

    c_optimizer.zero_grad()
    current_q = critic(state_batch, action_batch)
    next_actions = th.stack([actors_target[i](next_states_batch[:, i]) for i in range(n_agents)],
                            dim=-1)
    target_next_q = critic_target(next_states_batch, next_actions)
    target_q = target_next_q * gamma * (1 - done_batch[:, agent, :]) + reward_batch[:, agent, :]
    loss_q = mse_loss(current_q, target_q.detach())
    loss_q.backward()
    c_optimizer.step()

    a_optimizer.zero_grad()
    ac = action_batch.clone()
    ac[:, agent, :] = actor(state_batch[:, agent, :])
    loss_p = -critic(state_batch, ac).mean()
    loss_p.backward()
    a_optimizer.step()

    if update_target:
        soft_update(critic_target, critic, tau)
        soft_update(actors_target[agent], actor, tau)

    if queue is not None:
        queue.put([loss_q.item(), loss_p.item()])
    else:
        return loss_q.item(), loss_p.item()


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

        self.actors, self.critics = [], []
        self.actors_target, self.critics_target = [], []
        self.actors_optimizer, self.critics_optimizer = [], []
        for _ in range(n_agents):
            actor = Actor(dim_obs, dim_act)
            self.actors.append(actor)
            self.actors_target.append(deepcopy(actor))
            self.actors_optimizer.append(Adam(actor.parameters(), lr=args.a_lr))

            critic = Critic(n_agents, dim_obs, dim_act)
            self.critics.append(critic)
            self.critics_target.append(deepcopy(critic))
            self.critics_optimizer.append(Adam(critic.parameters(), lr=args.c_lr))
        self.mse_loss = nn.MSELoss()

        self.use_cuda = th.cuda.is_available()
        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()
            self.mse_loss.cuda()

        self.c_losses, self.a_losses = [], []
        self.step = 0
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
            net_visual([(1,) + dim_obs],
                       self.actors[0],
                       filename='actor',
                       directory=self.path['graph_path'],
                       format='png',
                       cleanup=True)
            self.actor_looker = NetLooker(net=self.actors[0],
                                          name='actor',
                                          is_look=False,
                                          root=self.path['graph_path'])
            net_visual([(1, n_agents,) + dim_obs, (1, n_agents, dim_act)],
                       self.critics[0],
                       filename='critic',
                       directory=self.path['graph_path'],
                       format='png',
                       cleanup=True)
            self.critic_looker = NetLooker(net=self.critics[0],
                                           name='critic',
                                           is_look=False)

    def add_experience(self, obs_n, act_n, next_obs_n, rew_n, done_n):
        obs_n = th.from_numpy(obs_n).type(FloatTensor)
        act_n = th.from_numpy(act_n).type(FloatTensor)
        next_obs_n = th.from_numpy(next_obs_n).type(FloatTensor)
        rew_n = th.from_numpy(rew_n).type(FloatTensor)
        done_n = th.from_numpy(done_n).type(FloatTensor)
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

    def update(self):
        self.step += 1
        num_threads = self.n_agents
        update_target = (self.step % 100 == 0)

        queue = Queue()
        workers = []
        for i in range(num_threads):
            work_args = (i, queue, num_threads,
                         self.actors[i], self.actors_target, self.critics[i], self.critics_target[i],
                         self.actors_optimizer[i], self.critics_optimizer[i], self.mse_loss,
                         self.tau, self.gamma,
                         self.memory,  self.batch_size,
                         update_target, )
            workers.append(Process(target=update_single_agent, args=work_args))
        for worker in workers:
            worker.start()

        c_loss, a_loss = [], []
        for _ in workers:
            [loss_q, lost_p] = queue.get()
            c_loss.append(loss_q)
            a_loss.append(lost_p)
        self.c_losses.append(c_loss)
        self.a_losses.append(a_loss)

        if update_target:
            # Record and visual the loss value of Actor and Critic
            self.scalars(key='critic_loss',
                         value={'agent_{}'.format(i + 1): v for i, v in enumerate(np.mean(self.c_losses, axis=0))},
                         episode=self.step)
            self.scalars(key='actor_loss',
                         value={'agent_{}'.format(i + 1): v for i, v in enumerate(np.mean(self.a_losses, axis=0))},
                         episode=self.step)
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
            raise NotImplementedError

    def save_model(self, save_path=None):
        if save_path is None:
            save_path = self.path['model_path']

        if save_path is not None:
            for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
                th.save(actor, save_path + 'actor_{}.pth'.format(i))
                th.save(critic, save_path + 'critic_{}.pth'.format(i))
        else:
            print('Save path is empty!')
            raise NotImplementedError

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
