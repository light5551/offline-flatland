import numpy as np
from flatland.envs.rail_env import RailEnv

env = RailEnv(width=30, height=30)
obs = env.reset()
while True:
    print(obs)
    obs, rew, done, info = env.step(
        {0: np.random.randint(0, 5), 1: np.random.randint(0, 5)}
    )
    if done:
        break
