import numpy as np
from math import inf
from typing import Dict, Optional, List, Tuple
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.observations import GlobalObsForRailEnv
from gymnasium import spaces, register, Env

class Obs(GlobalObsForRailEnv):
    def get_many(self, handles: Optional[List[int]] = None) -> Dict:
        obs = super().get_many(handles)
        for i, key in enumerate(obs):
            obs[key] = {
                "transition_map_array": np.array(obs[key][0]),
                "obs_agents_state": np.array(obs[key][1]),
                "obs_targets": np.array(obs[key][2])
            }
        return obs

class GymRailEnvWrapper(RailEnv, Env):
    def __init__(self, width, height, number_of_agents=2, obs_builder_object=Obs()):
        super().__init__(width=width, height=height, number_of_agents=number_of_agents, obs_builder_object=obs_builder_object)
        self.observation_space = spaces.Dict(
                {
                    # Observation space for Obs
                    i: spaces.Dict(
                        {
                            "transition_map_array": spaces.Box(-inf, inf, np.array([self.height, self.width, 16]), dtype=np.float64),
                            "obs_agents_state": spaces.Box(-inf, inf, np.array([self.height, self.width, 5]), dtype=np.float64),
                            "obs_targets": spaces.Box(-inf, inf, np.array([self.height, self.width, 2]), dtype=np.float64)
                        }
                    )
                    for i in range(self.number_of_agents)
                }
            )
        self.action_space = spaces.Dict(
            {
                i: spaces.Discrete(5) for i in range(self.number_of_agents)
            }
        )
        
    def reset(self, seed: int | None = None, options: dict| None = None) -> Tuple:
        return RailEnv.reset(self)
    
    def step(self, action_dict_: Dict[int, RailEnvActions]) -> Tuple:
        obs, rew, done, info = RailEnv.step(self, action_dict_)
        return obs, sum(rew.values()), done['__all__'], False, info # TODO: may be truncated should be fixed and reward should be changes

register("rail_env/flatland-v0.1", GymRailEnvWrapper)