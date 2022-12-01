import os
import time
import pickle
import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers

from algo import tf_util as U
from algo.trainer import MADDPGAgentTrainer
from env import AirSimDroneEnv

root = "/home/lydia/Documents/AirSim/"
setting_path = root + "settings.json"


def input_format(keyword, y_desc='Yes', n_desc='No'):
    while True:
        inputs = input("{} (Yes/yes/y/1):".format(keyword))
        if inputs in ['Yes', 'yes', 'y', '1']:
            print(y_desc)
            return True
        elif inputs in ['No', 'no', 'n', '0']:
            print(n_desc)
            return False
        else:
            print('Please input again')


def rewrite_setting_file(num_agents):
    print(setting_path, os.path)
    # Change the setting.json according to the number of UAV
    if os.path.exists(root):
        setting_str = r'{"SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",' \
                      r'"SettingsVersion": 1.2,' \
                      r'"SimMode": "Multirotor",' \
                      r'"ClockSpeed": 1,' \
                      r'"Vehicles":{'
        for i in range(num_agents):
            setting_str += '"Drone{}"'.format(i + 1) + r': {"VehicleType": "SimpleFlight",'
            setting_str += '"X": {}'.format((i + 1) * 10) + r', "Y": 10, "Z": -10}'
            if i != num_agents - 1:
                setting_str += ','
        setting_str += r'}}'
        print(setting_str)
        with open(setting_path, 'w') as f:
            f.write(setting_str)
        print('Writing successfully!\n')


def actor(inputs, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = inputs
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def critic(inputs, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = inputs
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def get_trainers(env, args):
    obs_shape_n = [env.observation_space.shape for _ in range(env.n)]
    act_shape_n = [env.action_space for _ in range(env.n)]
    trainers = []
    for i in range(env.n):
        trainers.append(
            MADDPGAgentTrainer(
                name="agent_%d" % i,
                actor=actor,
                critic=critic,
                obs_shape_n=obs_shape_n,
                act_space_n=act_shape_n,
                agent_index=i,
                args=args,
                local_q_func=args.good_policy == 'ddpg')
        )
    return trainers


def train(args):
    with U.single_threaded_session():
        # Create environment
        env = AirSimDroneEnv(
            ip_address="127.0.0.1",
            step_length=0.75,
            image_shape=(84, 84, 1),
            num_agents=args.num_agents,
        )
        # Create agent trainers
        trainers = get_trainers(env, args)
        print('Using good policy {} and adv policy {}'.format(args.good_policy, args.adv_policy))
        # Initialize
        U.initialize()
        if args.restore:
            print('Loading previous state...')
            U.load_state(args.load_dir or args.save_dir)
        # Parameter saver
        saver = tf.train.Saver()
        # Record
        rewards, losses = [], []  # sum of rewards for all agents
        episode_step = 0
        train_step = 0
        episode = 0

        print('Starting iterations...')
        t_start = time.time()
        obs_n = env.reset()
        while True:
            # get action
            # action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            action_n = [(np.random.random(env.action_space.shape[0])*2 - 1) for _ in trainers]
            print(train_step, action_n)
            # environment step
            new_obs_n, rew_n, done_n, _ = env.step(action_n)
            episode_step += 1
            train_step += 1
            terminal = episode_step >= args.max_episode_len
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i])
            obs_n = new_obs_n
            rewards.append([sum(rew_n), ] + rew_n)
            # update all trainers
            tmp = []
            for agent in trainers:
                loss = agent.update(trainers, train_step)
                if loss is not None:
                    tmp.append(loss)
            losses.append(tmp)
            # reset environment
            if any(done_n) or terminal:
                print(episode_step)
                obs_n = env.reset()
                episode_step = 0
                episode += 1
                if episode % args.save_rate == 0:  # save model, display training output
                    U.save_state(args.save_dir, saver=saver)
                    end = time.time()
                    mean_reward = np.array(rewards).mean(axis=0)
                    mean_loss = np.array(losses).mean(axis=0)
                    print("step: {}, episode: {}".format(train_step, episode))
                    print('mean reward: ', [round(v, 3) for v in mean_reward])
                    print('mean loss: ', [round(v, 3) for v in mean_loss])
                    print('time: ', round(end - t_start, 3))
                    t_start = end
                    rewards, losses = [], []
                if episode > args.num_episodes:  # end condition
                    break


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--num-agents", type=int, default=3, help="number of the agent (drone or car)")
    parser.add_argument("--max-episode-len", type=int, default=1000, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--learning-start", type=int, default=1000, help="start updating after this number of step")
    parser.add_argument("--good-policy", type=str, default="algo", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="algo", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./trained/policy/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default=None,
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--plots-dir", type=str, default="./trained/curve/",
                        help="directory where plot data is saved")
    args_ = parser.parse_args()
    if input_format(keyword='Rewrite the setting.json of Unreal project',
                    y_desc='The setting file is rewritten',
                    n_desc='The setting file is not changed'):
        rewrite_setting_file(num_agents=args_.num_agents)
    if input_format(keyword='The Unreal client has been opened',
                    y_desc='Start the training process',
                    n_desc='Not ready yet!'):
        train(args=args_)
