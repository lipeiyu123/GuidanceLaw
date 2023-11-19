from environment import MultiAgentEnv
from scenario import Scenario

def make_env():

    w = Scenario()
    world = w.make_world()
    env = MultiAgentEnv(world, w.reset_world, w.reward, w.observation,  w.has_winner)
    return env
