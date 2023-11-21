import gym
from gym import spaces
from gym.envs.registration import EnvSpec
from gym.utils import seeding
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
import random
import sys
sys.path.append('./Opt_mark1/')
from scenario import AD_END_DISTANCE


class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, done_callback=None):

        self.world = world

        self.landmarks = self.world.landmarks

        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = False
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = False               #############################
        self.time = 0

        ##############################################
        self.dt = 0.001
        self.ctrl_t = 0.01
        self.K = 5
        self.last_relative_position = self.agents[0].state.p_pos - self.landmarks[0].state.p_pos
        self.min_dist = 2500
        self.min_dist_list = []
        self.fov = []
        ##############################################

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p )
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,))
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,))
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = spaces.MultiDiscrete([[0,act_space.n-1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,),))
            agent.action.c = np.zeros(self.world.dim_c)
   


    def get_radir_detect(self ,self_p , self_angle , enemy_p , normalization_param =1 , noisy = False):

	###################################################
	###################################################
        return  dist_angle


    def Fun(self ,  input , other):
            out = np.zeros(4)
            out[0] = input[3] * math.cos(input[2])
            out[1] = input[3] * math.sin(input[2])
            out[2] = other
            out[3] = 0  
            return out

    def RK4(self , Fun , input , other , dt ):
            k1 = Fun(input , other)
            k2 = Fun(input + 0.5*dt*k1 , other)
            k3 = Fun(input + 0.5*dt*k2 , other)
            k4 = Fun(input + dt*k2 , other)
            input = input + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
            return input


    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = False      
        fun = self.Fun

	#########################################
	#...............................
	#########################################

        return obs_n, reward_n, done_n   



    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # record observations for each agent
        obs_n = []
        self.min_dist = 2500
        self.fov.clear
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self.get_obs(agent))
        return obs_n

    def get_attacker_pos (self):
        return self.landmarks[0].state.p_pos

   # get observation for a particular agent
    def get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    def get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)


    # set env action for a particular agent
    def set_action(self, action, agent, action_space):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, spaces.MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index + s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    # agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[0] += action[0][0]
                else:
                    agent.action.u = action[0]
            sensitivity = 1.25
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            # a = np.clip(np.random.normal(action[0], 2), *(-1, 1))  # add randomness to action selection for exploration
            action_s = np.clip(agent.action.u,*(-1,1))
            agent.action.u = action_s * np.pi / 9
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0, 'action len is not 2'

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, state , state_attacker):
        plt.plot(state[0][0] * 1000, state[0][1] * 1000, c='r', marker='o')
        plt.plot(state[1][0] * 1000, state[1][1] * 1000, c='g', marker='o')
        plt.plot(state_attacker[0], state_attacker[1], c='b', marker='o')
        plt.xlim(-5000,5000)
        plt.ylim(-5000,5000)
        # plt.show()
        plt.pause(0.01)

    def render_a(self):
        plt.plot( self.a_list)
        plt.pause(20)

    def save_min_distance(self,show = False):
        if show:
            self.min_dist_list.append(self.min_dist)
        print(self.min_dist  , "m\n")
        return self.min_dist
    
    def show_min_distance(self):
        plt.plot( self.min_dist_list)
        plt.ylabel("m")
        plt.xlabel("episodes")
        plt.xlim(0,1000)
        plt.pause(30)

    def show_fov(self):
        plt.plot( self.fov)
        plt.ylabel("Fov_degree")
        plt.xlabel("steps")
        plt.pause(10)
