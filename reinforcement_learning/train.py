import os

import numpy as np
import torch as th

from algo import Trainer
from env import AirSimDroneEnv
from parameters import ip_address

root = "/home/lydia/Documents/AirSim/"
setting_path = root + "settings.json"


def input_format(keyword, y_desc='Yes', n_desc='No'):
    while True:
        inputs = input("{} (Yes/yes/y/1):".format(keyword))
        if inputs in ['Yes', 'yes', 'y', '1']:
            print('\t--> YES:', y_desc)
            print('\t     NO:', n_desc, '\n')
            return True
        elif inputs in ['No', 'no', 'n', '0']:
            print('\t    YES:', y_desc)
            print('\t-->  NO:', n_desc, '\n')
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


def make_exp_id(args):
    return 'exp_{}_{}_{}_{}_{}_{}_{}'.format(args.exp_name, args.num_agents, args.seed,
                                             args.a_lr, args.c_lr, args.batch_size, args.gamma)


def train(args):
    # Seed
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    # Create environment
    env = AirSimDroneEnv(ip_address=ip_address,
                         step_length=0.75,
                         image_shape=(3, 144, 256),
                         num_agents=args.num_agents)
    # Create MARL trainer
    trainer = Trainer(env.n,
                      env.image_shape,
                      env.action_space.shape[0],
                      args,
                      folder=make_exp_id(args_))
    # Load previous param
    if args.load_dir is not None:
        trainer.load_model(load_path=args.load_dir)
    # Start iterations
    print('\n Iteration start...')
    step, episode, reward_record = 0, 0, []
    while True:
        episode += 1
        obs_n = env.reset()
        for i in range(args.max_episode_len):
            act_n = trainer.act(obs_n, step >= args.learning_start)
            next_obs_n, rew_n, done_n, _ = env.step(act_n)
            step += 1
            reward_record.append(rew_n)
            trainer.add_experience(obs_n, act_n, next_obs_n, rew_n, done_n)
            obs_n = next_obs_n
            print("{:>3d}, {:>5d}, {:>5d}".format(i, step, episode),
                  ["{:>+.3f}".format(a) for a in act_n.reshape(1, -1)[0]],
                  ["{:>+6.1f}".format(rew)for rew in rew_n], done_n)
            if step >= args.learning_start:
                trainer.update(step)

            if sum(done_n) > 0:
                break
        if episode % args.save_rate == 0:
            trainer.save_model()
            rew_dict = {'agent_{}'.format(i + 1): v for i, v in enumerate(np.mean(reward_record, axis=0))}
            rew_dict['total'] = np.sum(reward_record, axis=1).mean()
            trainer.scalars("Reward", rew_dict, episode)
            trainer.scalars("Param", {'var': trainer.var}, episode)
            reward_record = []
            print('Episode:', episode, rew_dict)
        if episode >= args.num_episodes:
            break
    # End environment
    env.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multi-agent environments")
    # Environment
    parser.add_argument("--num-agents", type=int, default=3, help="number of the agent (drone or car)")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument('--memory-length', default=int(1e6), type=int, help='number of experience replay pool')
    parser.add_argument("--learning-start", type=int, default=100, help="start updating after this number of step")
    parser.add_argument("--good-policy", type=str, default="algo", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="algo", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--a-lr", type=float, default=1e-4, help="learning rate for Actor Adam optimizer")
    parser.add_argument("--c-lr", type=float, default=1e-3, help="learning rate for Critic Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument('--tau', default=0.001, type=float, help='rate of soft update')
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='train', help="name of the experiment")
    parser.add_argument("--seed", type=int, default=1234, help="name of the experiment")
    parser.add_argument("--save-rate", type=int, default=100,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default=None,
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--plots-dir", type=str, default="./trained/curve/", help="directory where plot data is saved")

    args_ = parser.parse_args()
    if input_format(keyword='Rewrite the setting.json of Unreal project',
                    y_desc='The setting file is rewritten',
                    n_desc='The setting file is not changed'):
        rewrite_setting_file(num_agents=args_.num_agents)

    if input_format(keyword='The Unreal client has been opened',
                    y_desc='Execute the train function',
                    n_desc='Not ready yet!'):
        train(args=args_)
