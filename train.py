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
#from parl.algorithms.paddle import MADDPG
from maddpg import MADDPG
from parl.utils import logger, summary
import make_env
import matplotlib.pyplot as plt
import paddle


CRITIC_LR = 0.0005 # learning rate for the critic model
ACTOR_LR = 0.00005  # learning rate of the actor model
GAMMA = 0.998  # reward discount factor
TAU = 0.01  # soft update
BATCH_SIZE = 32768
MAX_STEP_PER_EPISODE = 1024  # maximum step per episode
EVAL_EPISODES = 10
USE_THREAD = False


# Runs policy and returns episodes' rewards and steps for evaluation
def run_evaluate_episodes(env, agents, eval_episodes):
    eval_episode_rewards = []
    eval_episode_steps = []
    done_list = []
    if args.show:
        plt.close()
        plt.figure()
    while len(eval_episode_rewards) < eval_episodes:
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
            if done or steps >= MAX_STEP_PER_EPISODE-1:
                done_list.append(done)

            total_reward += sum(reward_n)
            # show animation
            if args.show and steps % 10 == 0 :
                env.render(obs_n , attacker_pos)
                plt.clf()
        eval_min_distance = env.save_min_distance()
        eval_episode_rewards.append(total_reward)
        eval_episode_steps.append(steps)
    return eval_episode_rewards, eval_episode_steps , eval_min_distance , done_list


def run_episode(env, agents):
    act_flag = False
    cri_flag = False
    obs_n = env.reset()
    done = False
    total_reward = 0
    agents_reward = [0 for _ in range(env.n)]
    steps = 0
    if args.show:
        plt.close()
        plt.figure()
    while not done and steps < MAX_STEP_PER_EPISODE:
        steps += 1
        action_n = [agent.sample(obs) for agent, obs in zip(agents, obs_n)]
        next_obs_n, reward_n, done_n= env.step(action_n)
        attacker_pos = env.get_attacker_pos()
        done = any(done_n)

        # store experience
        for i, agent in enumerate(agents):
            agent.add_experience(obs_n[i], action_n[i], reward_n[i],next_obs_n[i], done_n[i])

        # compute reward of every agent
        obs_n = next_obs_n
        for i, reward in enumerate(reward_n):
            total_reward += reward
            agents_reward[i] += reward

        # show model effect without training
        if args.restore and args.show:
            continue
    
        if args.show and steps % 10 == 0:
            env.render(obs_n , attacker_pos)
            plt.clf()
            

        # learn policy
        for i, agent in enumerate(agents):
            loss = agent.learn(agents)
            if  loss: 
                if i == 0:
                    t_critic_loss = float(loss[1].cpu().detach())
                    t_actor_loss = float(loss[0].cpu().detach())
                    act_flag = True
                if i == 1:
                    d_critic_loss = float(loss[1].cpu().detach())
                    d_actor_loss = float(loss[0].cpu().detach())   
                    cri_flag = True
        if  act_flag and cri_flag:
            logger.info('t_critic_loss {}, t_actor_loss {}, d_critic_loss {}, d_actor_loss {}'.format(t_critic_loss, t_actor_loss , d_critic_loss , d_actor_loss))
            act_flag = False
            cri_flag = False
    min_distance = env.save_min_distance()

    return total_reward, agents_reward, steps ,min_distance


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
            model_file = args.model_dir + '/agent_' + str(i)   + "_rew_1.7068_episode_49000"
            if not os.path.exists(model_file):
                raise Exception(
                    'model file {} does not exits'.format(model_file))
            agents[i].restore(model_file)


    total_steps = 0
    total_episodes = 0
    #while total_episodes <= args.max_episodes:
    while True:
        # run an episode
        ep_reward, ep_agent_rewards, steps , min_distance = run_episode(env, agents)
        summary.add_scalar('train/episode_reward_wrt_episode', ep_reward,
                           total_episodes)
        summary.add_scalar('train/episode_reward_wrt_step', ep_reward,
                           total_steps)
        logger.info(
            'total_steps {}, episode {}, reward {}, agents rewards {}, episode steps {}, min_distance {}'
            .format(total_steps, total_episodes, ep_reward, ep_agent_rewards,
                    steps, min_distance ))

        total_steps += steps
        total_episodes += 1

        # evaluste agents
        if total_episodes % args.test_every_episodes == 0:

            eval_episode_rewards, eval_episode_steps , eval_min_distance , done_list = run_evaluate_episodes(
                env, agents, EVAL_EPISODES)

            eps = np.mean(eval_episode_rewards)
            summary.add_scalar('eval/episode_reward',
                               np.mean(eval_episode_rewards), total_episodes)
            logger.info('Evaluation over: {} episodes, Reward: {}, eval_min_distance {}'.format(
                EVAL_EPISODES, eps, eval_min_distance))

            # save model
            #if not args.restore and  eps >= 280:
            print(done_list)
            print(all(done_list))
            if eps >= 1.2 and all(done_list) :
                
                model_dir = args.model_dir
                os.makedirs(os.path.dirname(model_dir), exist_ok=True)
                for i in range(len(agents)):
                    model_name = '/agent_' + str(i) +"_rew_" + str(round((np.mean(eval_episode_rewards) ) , 4)) + "_episode_" + str(total_episodes) 
                    agents[i].save(model_dir + model_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument('--show',                                    action='store_true',           default=False,               help='display or not')
    parser.add_argument('--restore',                                action='store_true',            default=False,              help='restore or not, must have model_dir')
    parser.add_argument('--model_dir',                          type=str,                                  default='./model',      help='directory for saving model')
    parser.add_argument(   '--continuous_actions',   action='store_true',            default=True,               help='use continuous action mode or not')
    parser.add_argument(   '--max_episodes',              type=int,                                   default=100000,            help='stop condition: number of episodes')
    parser.add_argument(  '--test_every_episodes',  type=int,                                    default=100,       help='the episode interval between two consecutive evaluations')
    args = parser.parse_args()
    
    main()
