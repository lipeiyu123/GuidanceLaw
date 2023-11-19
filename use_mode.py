#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import parl
import threading
import os
import time
import argparse
import numpy as np
from simple_model import MAModel
from simple_agent import MAAgent
from parl.algorithms.paddle import MADDPG
from parl.utils import logger, summary
import make_env
import matplotlib.pyplot as plt

CRITIC_LR = 0.001  # learning rate for the critic model
ACTOR_LR = 0.001  # learning rate of the actor model
GAMMA = 0.998  # reward discount factor
TAU = 0.01  # soft update
BATCH_SIZE = 512
MAX_STEP_PER_EPISODE = 1024  # maximum step per episode
EVAL_EPISODES = 100
USE_THREAD = False



# Runs policy and returns episodes' rewards and steps for evaluation
def run_evaluate_episodes(env, agents, eval_episodes):
    eval_episode_rewards = []
    eval_episode_steps = []
    succeed_times = 0
    if args.show:
        plt.close()
        plt.figure()
    while len(eval_episode_rewards) < eval_episodes:
        print(len(eval_episode_rewards)+1 , "eps\n")
        obs_n = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < MAX_STEP_PER_EPISODE:
            steps += 1
            action_n = [
                agent.predict(obs) for agent, obs in zip(agents, obs_n)
            ]
            obs_n, reward_n, done_n = env.step(action_n)
            attacker_pos = env.get_attacker_pos()
            done = any(done_n)
            if done_n[1] :
                succeed_times +=1
            total_reward += sum(reward_n)
            # show animation
            if args.show and steps % 10 == 0 :
                env.render(obs_n , attacker_pos)
                plt.clf()
        env.save_min_distance(show = True)
        eval_episode_rewards.append(total_reward)
        eval_episode_steps.append(steps)
    env.show_min_distance()
    #env.show_fov()
    #env.render_a()
    print("succeed rate : " ,  (succeed_times /  EVAL_EPISODES) * 100 , "%")
    return eval_episode_rewards, eval_episode_steps


def main():
    env = make_env.make_env()
    #critic_in_dim = sum(env.observation_space) + sum(env.action_space)
    critic_in_dim = 0
    n = env.n
    obs_space = []
    act_space = []
    last_eval_episode_rewards = 0
    for i in range(n):
        critic_in_dim = critic_in_dim + env.action_space[i].shape[0]
        act_space.append(env.action_space[i].shape[0])
    for i in range(n):
        critic_in_dim = critic_in_dim + env.observation_space[i].shape[0]
        obs_space.append(env.observation_space[i].shape[0])
    # build agents
    agents = []
    
    for i in range(env.n):
        model = MAModel(env.observation_space[i].shape[0], env.action_space[i].shape[0], critic_in_dim,
                        args.continuous_actions)
        algorithm = MADDPG(
            model,
            agent_index=i,
            act_space=env.action_space,
            gamma=GAMMA,
            tau=TAU,
            critic_lr=CRITIC_LR,
            actor_lr=ACTOR_LR)
        agent = MAAgent(
            algorithm,
            agent_index=i,
            obs_dim_n=obs_space,
            act_dim_n=act_space,
            batch_size=BATCH_SIZE)
        agents.append(agent)

    if args.restore:
        # restore modle
        for i in range(len(agents)):
            model_file = args.model_dir + '/agent_' + str(i)   + "_rew_1.905_episode_34300"
            if not os.path.exists(model_file):
                raise Exception(
                    'model file {} does not exits'.format(model_file))
            agents[i].restore(model_file)


    eval_episode_rewards, eval_episode_steps = run_evaluate_episodes(
        env, agents, EVAL_EPISODES)
    eps = np.mean(eval_episode_rewards)
    #logger.info('Evaluation over: {} episodes, Reward: {}'.format(
    #   EVAL_EPISODES, eps))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument('--show',                                    action='store_true',           default=False,               help='display or not')
    parser.add_argument('--restore',                                action='store_true',            default=True,              help='restore or not, must have model_dir')
    parser.add_argument('--model_dir',                          type=str,                                  default='./model',      help='directory for saving model')
    parser.add_argument(   '--continuous_actions',   action='store_true',            default=True,               help='use continuous action mode or not')
    parser.add_argument(   '--max_episodes',              type=int,                                   default=300,            help='stop condition: number of episodes')
    parser.add_argument(  '--test_every_episodes',  type=int,                                    default=50,       help='the episode interval between two consecutive evaluations')
    #parser.add_argument(   '--num',                            type=int,                                   default=0,            help='stop condition: number of episodes')
    args = parser.parse_args()

    main()
